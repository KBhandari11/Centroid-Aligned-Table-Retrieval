import os, sys
import json
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from utils.cache_embedding import cache_exists_for_model, load_cache_for_model
from utils.core import SPLADE_TOPK, _topk_dense_to_sparse
from utils.Adapter import (
    HierDenseAdapter, HierSparseAdapter,
    REPRESENTATIVE_ORDER, CATEGORIES, CAT_NAMES, FMT2IDX, FMT2CAT, N_FMT, N_CAT,
)
from utils.TrainerConfig import HierCfg 

# =============================================================================
# Chunk aggregation and aligned cache loading (unchanged behaviour)
# =============================================================================
def _aggregate_chunks(doc_ids: List[str], emb, model_name: str) -> Dict[str, object]:
    import scipy.sparse
    table_to_rows: Dict[str, List[int]] = defaultdict(list)
    for i, did in enumerate(doc_ids):
        table_to_rows[did.rsplit("::", 1)[0]].append(i)
    result = {}
    if model_name == "splade":
        V = emb.shape[1]
        for tid, rows in table_to_rows.items():
            avg = np.asarray(emb[rows].mean(axis=0), dtype=np.float32).ravel()
            nz = np.flatnonzero(avg > 0)
            if len(nz) > SPLADE_TOPK:
                nz = nz[np.argpartition(avg[nz], -SPLADE_TOPK)[-SPLADE_TOPK:]]
            result[tid] = scipy.sparse.csr_matrix(
                (avg[nz].tolist(), ([0] * len(nz), nz.tolist())), shape=(1, V), dtype=np.float32)
    else:
        for tid, rows in table_to_rows.items():
            result[tid] = np.asarray(emb[np.array(rows)], dtype=np.float32).mean(axis=0)
    return result


def load_rep_caches_aligned(cache_root, model_name, dataset, rep_names, cache_prefix="chunked_"):
    import scipy.sparse
    rep_to_emb, rep_to_row, D = {}, {}, None
    for r in rep_names:
        key = f"{cache_prefix}{r}"
        if not cache_exists_for_model(cache_root, model_name, dataset, key):
            continue
        doc_ids, emb = load_cache_for_model(cache_root, model_name, dataset, key)
        if emb is None or len(doc_ids) == 0:
            continue
        if cache_prefix:
            table_to_emb = _aggregate_chunks(doc_ids, emb, model_name)
            tids = sorted(table_to_emb)
            if model_name == "splade":
                emb_agg = scipy.sparse.vstack([table_to_emb[t] for t in tids], format="csr")
            else:
                emb_agg = np.stack([table_to_emb[t] for t in tids], axis=0)
        else:
            tids = list(doc_ids)
            emb_agg = emb.tocsr() if scipy.sparse.issparse(emb) else np.asarray(emb, np.float32)
        d = emb_agg.shape[1]
        if D is None:
            D = d
        elif d != D:
            raise ValueError(f"Embedding dim mismatch for rep={key}: got {d}, expected {D}")
        rep_to_emb[r] = emb_agg
        rep_to_row[r] = {t: i for i, t in enumerate(tids)}
    if D is None or len(rep_to_emb) < 2:
        raise ValueError(f"[{dataset}] Need at least 2 cached reps. Found: {list(rep_to_emb)}")
    counts = defaultdict(int)
    for m in rep_to_row.values():
        for t in m:
            counts[t] += 1
    table_ids = sorted(t for t, c in counts.items() if c >= 2)
    return rep_to_emb, rep_to_row, table_ids, D


# =============================================================================
# Geometry helpers (dot product world: raw means, relative squared distances)
# =============================================================================
def _row(emb, i, is_sparse) -> np.ndarray:
    if is_sparse:
        return np.asarray(emb[i].toarray(), dtype=np.float32).ravel()
    return np.asarray(emb[i], dtype=np.float32)


def _resparsify(v: np.ndarray, topk: int) -> np.ndarray:
    out = np.zeros_like(v)
    nz = np.flatnonzero(v > 0)
    if len(nz) > topk:
        nz = nz[np.argpartition(v[nz], -topk)[-topk:]]
    out[nz] = v[nz]
    return out


def rel_sq_dist_np(a: np.ndarray, b: np.ndarray, eps=1e-8) -> float:
    return float(((a - b) ** 2).sum() / ((a ** 2).sum() + eps))


def rel_sq_dist(a: torch.Tensor, b: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """||a - b||^2 / ||a||^2  (row-wise). Scale free; consistent with dot-product scoring."""
    return ((a - b) ** 2).sum(-1) / (a.pow(2).sum(-1) + eps)


def table_views(rep_to_emb, rep_to_row, tid, is_sparse) -> Dict[str, np.ndarray]:
    return {f: _row(rep_to_emb[f], rep_to_row[f][tid], is_sparse)
            for f in REPRESENTATIVE_ORDER if f in rep_to_row and tid in rep_to_row[f]}


def targets_for_table(views: Dict[str, np.ndarray], is_sparse: bool):
    """Returns c_all (raw mean) and per-view leave-one-out category centroids."""
    stack = np.stack(list(views.values()), 0)
    c_all = stack.mean(0)
    if is_sparse:
        c_all = _resparsify(c_all, SPLADE_TOPK)
    c_cat = {}
    for f in views:
        cat = CAT_NAMES[FMT2CAT[f]]
        sib = [views[g] for g in CATEGORIES[cat] if g in views and g != f]
        cc = np.stack(sib, 0).mean(0) if sib else c_all
        if is_sparse and sib:
            cc = _resparsify(cc, SPLADE_TOPK)
        c_cat[f] = cc
    return c_all, c_cat


def estimate_format_stats(rep_to_emb, rep_to_row, table_ids, is_sparse, n_sample):
    """
    mu_delta[f]  : mean over tables of (e_f(T) - c_all(T))        -> dense bias init
    log_gate[f]  : log( mean_T c_all / mean_T e_f ) per term        -> sparse gate init
    tau[f]       : median_T ||e_f - c_all||^2 / ||e_f||^2           -> displacement bound
    """
    D = next(iter(rep_to_emb.values())).shape[1]
    rng = random.Random(0)
    sample = rng.sample(table_ids, min(n_sample, len(table_ids)))
    delta_sum = np.zeros((N_FMT, D), np.float64)
    e_sum = np.zeros((N_FMT, D), np.float64)
    c_sum = np.zeros(D, np.float64)
    n = np.zeros(N_FMT, np.int64)
    dists = [[] for _ in range(N_FMT)]
    for tid in sample:
        views = table_views(rep_to_emb, rep_to_row, tid, is_sparse)
        if len(views) < 2:
            continue
        c_all, _ = targets_for_table(views, is_sparse)
        c_sum += c_all
        for f, v in views.items():
            i = FMT2IDX[f]
            delta_sum[i] += v - c_all
            e_sum[i] += v
            n[i] += 1
            dists[i].append(rel_sq_dist_np(v, c_all))
    n_safe = np.maximum(n, 1)[:, None]
    mu_delta = (delta_sum / n_safe).astype(np.float32)
    c_mean = c_sum / max(1, len(sample))
    log_gate = np.log((c_mean[None, :] + 1e-3) / (e_sum / n_safe + 1e-3)).astype(np.float32)
    log_gate[n == 0] = 0.0
    tau = np.array([float(np.median(d)) if d else 0.0 for d in dists], np.float32)
    for f in REPRESENTATIVE_ORDER:
        i = FMT2IDX[f]
        if n[i]:
            print(f"[stats] {f:16s} n={n[i]:6d} ||mu_delta||={np.linalg.norm(mu_delta[i]):.4f} tau={tau[i]:.4f}", flush=True)
    return mu_delta, log_gate, tau


# =============================================================================
# Dataset: one item = one table, emits every available view with its targets
# =============================================================================
class HierTableDataset(Dataset):
    def __init__(self, dataset_to_rep_to_emb, dataset_to_rep_to_row, samples, tau, is_sparse, rho):
        self.E, self.R = dataset_to_rep_to_emb, dataset_to_rep_to_row
        self.samples, self.tau, self.sparse, self.rho = samples, tau, is_sparse, rho

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ds, tid = self.samples[idx]
        views = table_views(self.E[ds], self.R[ds], tid, self.sparse)
        c_all, c_cat = targets_for_table(views, self.sparse)
        out = []
        for f, e in views.items():
            cc = c_cat[f]
            out.append(dict(
                table=idx, fmt=FMT2IDX[f], e=e, c_all=c_all, c_cat=cc,
                d_cat_all=rel_sq_dist_np(cc, c_all),
                tau=self.rho * float(self.tau[FMT2IDX[f]]),
            ))
        return out


def hier_collate(batch):
    flat = [v for views in batch for v in views]
    out = {}
    for k in flat[0]:
        vals = [b[k] for b in flat]
        out[k] = torch.from_numpy(np.stack(vals, 0)) if isinstance(vals[0], np.ndarray) \
            else torch.as_tensor(vals)
    return out


# =============================================================================
# Loss
# =============================================================================
def hier_loss(z, e, logits, fmt, table, c_all, c_cat, d_cat_all, tau, cfg: HierCfg,
              step, total_steps, expand=None):
    t = min(1.0, step / max(1, cfg.anneal_frac * total_steps))
    w_cat = cfg.w_cat * (1.0 - 0.75 * t)
    w_all = cfg.w_all * (0.25 + 0.75 * t)

    d_all = rel_sq_dist(c_all, z)        # relative to target norm
    d_cat = rel_sq_dist(c_cat, z)
    d_id = rel_sq_dist(e, z)             # relative to frozen norm

    L_all = F.relu(d_all - cfg.delta_all).mean()
    L_cat = F.relu(d_cat - cfg.delta_cat).mean()
    L_prog = F.relu(d_all - d_cat_all + cfg.margin).mean()   # beat your own family centroid
    L_bound = F.relu(d_id - tau).mean()                       # bounded displacement

    # dot-product separation, scaled by frozen norms so it is scale free
    scale = e.norm(dim=-1)[:, None] * e.norm(dim=-1)[None, :] + 1e-8
    S_z, S_e = (z @ z.T) / scale, (e @ e.T) / scale
    diff = (table[:, None] != table[None, :]).float()
    L_sep = (F.relu(S_z - S_e) * diff).sum() / diff.sum().clamp_min(1.0)

    L_fmt = F.cross_entropy(logits, fmt)
    L_flops = (expand.mean(0) ** 2).sum() if expand is not None else z.new_zeros(())

    loss = (w_all * L_all + w_cat * L_cat + cfg.w_prog * L_prog + cfg.w_bound * L_bound
            + cfg.w_sep * L_sep + cfg.w_fmt * L_fmt + cfg.w_flops * L_flops)
    logs = dict(L_all=L_all.item(), L_cat=L_cat.item(), L_prog=L_prog.item(),
                L_bound=L_bound.item(), L_sep=L_sep.item(), L_fmt=L_fmt.item(),
                L_flops=float(L_flops), d_all=d_all.mean().item(), d_id=d_id.mean().item(),
                w_all=w_all, w_cat=w_cat)
    return loss, logs


# =============================================================================
# Training
# =============================================================================
def train(cache_root, model_name, datasets, reps, out_path, cfg: HierCfg):
    is_sparse = (model_name == "splade")
    E, R, samples, D = {}, {}, [], None
    for ds in datasets:
        rep_to_emb, rep_to_row, table_ids, D_ds = load_rep_caches_aligned(
            cache_root, model_name, ds, reps, cfg.cache_prefix)
        if D is None:
            D = D_ds
        elif D_ds != D:
            raise ValueError(f"Embedding dim mismatch across datasets: {ds} has {D_ds}, expected {D}")
        E[ds], R[ds] = rep_to_emb, rep_to_row
        samples.extend((ds, t) for t in table_ids)
        print(f"[{ds}] tables={len(table_ids)} reps={list(rep_to_emb)}")
    if not samples:
        raise ValueError("No training tables found.")
    print("All the embedding loaded...", flush=True)
    # closed-form format statistics (pooled across datasets)
    E_pool = {f: None for f in reps}
    mu_parts, lg_parts, tau_parts, n_parts = [], [], [], []
    for ds in datasets:
        tids = [t for d, t in samples if d == ds]
        mu, lg, tau = estimate_format_stats(E[ds], R[ds], tids, is_sparse, cfg.stat_sample)
        mu_parts.append(mu); lg_parts.append(lg); tau_parts.append(tau); n_parts.append(len(tids))
    w = np.array(n_parts, np.float64) / sum(n_parts)
    mu_delta = sum(wi * m for wi, m in zip(w, mu_parts)).astype(np.float32)
    log_gate = sum(wi * g for wi, g in zip(w, lg_parts)).astype(np.float32)
    tau = sum(wi * t for wi, t in zip(w, tau_parts)).astype(np.float32)

    # balanced sampling across datasets (NQ would otherwise be ~96% of samples)
    ds_count = defaultdict(int)
    for d, _ in samples:
        ds_count[d] += 1
    weights = [1.0 / ds_count[d] for d, _ in samples]
    sampler = WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)

    ds_obj = HierTableDataset(E, R, samples, tau, is_sparse, cfg.rho)
    dl = DataLoader(ds_obj, batch_size=cfg.tables_per_batch, sampler=sampler,
                    num_workers=cfg.num_workers, collate_fn=hier_collate, drop_last=True)

    if is_sparse:
        adapter = HierSparseAdapter(D, r=cfg.r, dropout=cfg.dropout, k_exp=cfg.k_exp,
                                    topk=SPLADE_TOPK, log_gate_init=log_gate)
        offset_params = [adapter.log_gate]
    else:
        adapter = HierDenseAdapter(D, r=cfg.r, dropout=cfg.dropout, mu_delta=mu_delta)
        offset_params = [adapter.bias]
    adapter = adapter.to(cfg.device)
    offset_ids = {id(p) for p in offset_params}
    other = [p for p in adapter.parameters() if id(p) not in offset_ids]
    opt = torch.optim.AdamW([
        {"params": other, "lr": cfg.lr},
        {"params": offset_params, "lr": cfg.lr * cfg.bias_lr_mult, "weight_decay": 0.0},
    ], weight_decay=cfg.weight_decay)
    print(adapter)
    print(f"Dimension: {D}  sparse: {is_sparse}")

    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    traj_jsonl, traj_pkl = os.path.join(out_dir, "trajectory.jsonl"), os.path.join(out_dir, "trajectory.pkl")
    meta = dict(type="meta", model_name=model_name, datasets=datasets, reps=reps, D=D,
                cfg=asdict(cfg), scoring="dot_product")
    with open(traj_jsonl, "a") as f:
        f.write(json.dumps(meta) + "\n")
    trajectory = []

    def _save(path):
        torch.save(dict(state_dict=adapter.state_dict(), D=D, model_name=model_name,
                        datasets=datasets, reps=reps, cfg=asdict(cfg), scoring="dot_product",
                        mu_delta=mu_delta, log_gate=log_gate, tau=tau), path)

    adapter.train()
    it = iter(dl)
    for step in range(1, cfg.steps + 1):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl)
            b = next(it)
        e = b["e"].to(cfg.device).float()
        fmt = b["fmt"].to(cfg.device)
        if is_sparse:
            z, logits, expand = adapter(e, fmt)
        else:
            z, logits = adapter(e, fmt)
            expand = None
        loss, logs = hier_loss(
            z, e, logits, fmt, b["table"].to(cfg.device),
            b["c_all"].to(cfg.device).float(), b["c_cat"].to(cfg.device).float(),
            b["d_cat_all"].to(cfg.device).float(), b["tau"].to(cfg.device).float(),
            cfg, step, cfg.steps, expand=expand)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), cfg.clip)
        opt.step()

        if step % cfg.log_every == 0:
            with torch.no_grad():
                rec = dict(type="log", step=step, loss=float(loss.item()), **logs,
                           mean_norm_e=e.norm(dim=-1).mean().item(),
                           mean_norm_z=z.norm(dim=-1).mean().item(),
                           dot_ze_over_ee=((z * e).sum(-1) / (e * e).sum(-1).clamp_min(1e-8)).mean().item(),
                           fmt_acc=(logits.argmax(-1) == fmt).float().mean().item(),
                           nnz_z=float((z > 0).float().sum(-1).mean().item()) if is_sparse else 0.0,
                           unique_tables_in_batch=int(torch.unique(b["table"]).numel()))
            trajectory.append(rec)
            with open(traj_jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{model_name}] step={step} loss={rec['loss']:.4f} all={rec['L_all']:.4f} "
                  f"cat={rec['L_cat']:.4f} prog={rec['L_prog']:.4f} bound={rec['L_bound']:.4f} "
                  f"sep={rec['L_sep']:.4f} fmt={rec['L_fmt']:.3f}({rec['fmt_acc']:.2f}) "
                  f"d_all={rec['d_all']:.4f} d_id={rec['d_id']:.4f} "
                  f"|e|={rec['mean_norm_e']:.3f} |z|={rec['mean_norm_z']:.3f}"
                  + (f" nnz={rec['nnz_z']:.0f}" if is_sparse else ""),flush=True)

        if step % cfg.ckpt_every == 0:
            _save(out_path)

    _save(out_path)
    print(f"Saved adapter to: {out_path}")
    with open(traj_pkl, "wb") as f:
        pickle.dump(dict(meta=meta, trajectory=trajectory), f)


# =============================================================================
# Inference helper (dot-product space; no normalisation)
# =============================================================================
@torch.no_grad()
def centroidize_embedding(adapter, e: np.ndarray, model_name="dense", fmt_name: Optional[str] = None,
                          device: Optional[str] = None):
    """
    fmt_name: serialization used to produce e. Pass None for mixed/unknown format,
    in which case the adapter soft-selects the offset from its format head.
    """
    if device is None:
        device = next(adapter.parameters()).device
    adapter.eval()
    t = torch.from_numpy(np.asarray(e, dtype=np.float32)).to(device)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    fmt = None if fmt_name is None else torch.full((t.shape[0],), FMT2IDX[fmt_name], device=device, dtype=torch.long)
    z = adapter(t, fmt)[0].cpu().numpy().astype(np.float32)
    if model_name == "splade":
        return _topk_dense_to_sparse(z, SPLADE_TOPK)
    return z[0] if z.shape[0] == 1 else z


def load_adapter(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    cfg = HierCfg(**ck["cfg"])
    if ck["model_name"] == "splade":
        a = HierSparseAdapter(ck["D"], r=cfg.r, dropout=cfg.dropout, k_exp=cfg.k_exp, topk=SPLADE_TOPK)
    else:
        a = HierDenseAdapter(ck["D"], r=cfg.r, dropout=cfg.dropout)
    a.load_state_dict(ck["state_dict"])
    return a.to(device).eval()


# =============================================================================
# CLI (same as before): python -m ...train_universal_adapter <model> <date> <subset|all> [cache_prefix]
# =============================================================================
if __name__ == "__main__":
    cache_root = "/data/Kushal/UniversalRetrieval_data/emb_cache"
    model_name = sys.argv[1]
    date = sys.argv[2]
    subset = sys.argv[3]
    cache_prefix = sys.argv[4] if len(sys.argv) > 4 else "chunked_"
    is_sparse = (model_name == "splade")

    cfg = HierCfg(cache_prefix=cache_prefix)

    if subset == "subset":
        datasets = ["WTQ", "WIKISQL"]
        out_path = f"/data/Kushal/UniversalRetrieval_data/{date}/centroid_adapter_subset_dataset/{model_name}/adapter.pt"
    else:
        datasets = ["NQ", "WTQ", "WIKISQL"]
        out_path = f"/data/Kushal/UniversalRetrieval_data/{date}/centroid_adapter/{model_name}/adapter.pt"

    print(f"Datasets: {datasets}  model: {model_name}  SPLADE: {is_sparse}  scoring: dot product")
    train(cache_root=cache_root, model_name=model_name, datasets=datasets,
          reps=REPRESENTATIVE_ORDER, out_path=out_path, cfg=cfg)
    
#CUDA_VISIBLE_DEVICES=2 python -m adapter.train_universal_adapter splade 2026-09-17 all > ./log/training_splade_rank_chunked_new.txt 2>&1
#CUDA_VISIBLE_DEVICES=2 python -m adapter.train_universal_adapter bge 2026-09-17 all > ./log/training_bge_rank_chunked_new.txt 2>&1
#CUDA_VISIBLE_DEVICES=2 python -m adapter.train_universal_adapter mdbr 2026-09-12 subset > ./log/training_subset_mdbr_rank_chunked.txt 2>&1
#CUDA_VISIBLE_DEVICES=2 python -m adapter.train_universal_adapter miniLM 2026-09-17 all > ./log/training_miniLM_rank_chunked_new.txt 2>&1
#CUDA_VISIBLE_DEVICES=2 python -m adapter.train_universal_adapter mpnet 2026-09-12 subset > ./log/training_subset_mpnet_rank_chunked.txt 2>&1
#CUDA_VISIBLE_DEVICES=2 python -m adapter.train_universal_adapter qwen0.6B 2026-09-12 subset > ./log/training_subset_qwen0.6B_rank_chunked.txt 2>&1
#CUDA_VISIBLE_DEVICES=2 python -m adapter.train_universal_adapter reasonir 2026-09-12 subset > ./log/training_subset_reasonir_rank_chunked.txt 2>&1
#CUDA_VISIBLE_DEVICES=2 python -m adapter.train_universal_adapter qwen4B 2026-09-12 subset > ./log/training_subset_qwen4B_rank_chunked.txt 2>&1
