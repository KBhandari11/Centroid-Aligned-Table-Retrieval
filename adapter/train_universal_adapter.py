import os, sys
import random

from typing import Dict, List, Tuple, Optional

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from ..utils.cache_embedding import cache_exists, load_cache_mmap
from ..utils.Adapter import ResidualCentroidAdapter,BottleneckResidualAdapter

from ..utils.losses import invariance_to_centroid_loss,  variance_loss, covariance_loss, identity_preservation_loss
from ..utils.dataset import CacheMultiViewDatasetGiant, collate_fn
from ..utils.TrainerConfig import TrainCfg
# -----------------------------
# Representation groups
# -----------------------------
REPRESENTATIVE_ORDER = [
    'pipe_serialized', 'token_serialized', 'space_serialized',
    'csv', 'tsv', 'html', 'markdown', 'latex', 'dict', 'json', 'xml',
    'shuffled_rows', 'shuffled_cols', 'transpose',
    'mschema', 'macschema', 'ddl'
]


def print_cache_norm_stats(rep_to_emb, n=5000):
    # sample from the first rep we have
    r0 = next(iter(rep_to_emb.keys()))
    emb = rep_to_emb[r0]
    N = min(len(emb), n)
    idx = np.random.choice(len(emb), size=N, replace=False)
    x = np.asarray(emb[idx], dtype=np.float32)
    norms = np.linalg.norm(x, axis=1)
    print(f"[cache norms] rep={r0} N={N} mean={x.mean():.4f} | {norms.mean():.4f} std={x.std():.4f} | {norms.std():.4f} "
          f"min={x.min():.4f} | {norms.min():.4f} max={x.max():.4f} | {norms.max():.4f}")




# -----------------------------
# Cache loader: align by table_id across reps (PER DATASET)
# -----------------------------
def load_rep_caches_aligned(
    cache_root: str,
    model_name: str,
    dataset: str,
    rep_names: List[str],
) -> Tuple[Dict[str, np.memmap], Dict[str, Dict[str, int]], List[str], int]:
    rep_to_emb: Dict[str, np.memmap] = {}
    rep_to_row: Dict[str, Dict[str, int]] = {}
    D = None

    for r in rep_names:
        if not cache_exists(cache_root, model_name, dataset, r):
            continue
        doc_ids, emb = load_cache_mmap(cache_root, model_name, dataset, r)
        if emb is None or len(doc_ids) == 0:
            continue

        if D is None:
            D = emb.shape[1]
        elif emb.shape[1] != D:
            raise ValueError(f"Embedding dim mismatch for rep={r}: got {emb.shape[1]} expected {D}")

        rep_to_emb[r] = emb
        rep_to_row[r] = {str(tid): i for i, tid in enumerate(doc_ids)}

    if D is None or len(rep_to_emb) < 2:
        raise ValueError(f"[{dataset}] Need at least 2 cached reps with embeddings to train centroid adapter.")

    counts: Dict[str, int] = {}
    for mapping in rep_to_row.values():
        for tid in mapping.keys():
            counts[tid] = counts.get(tid, 0) + 1

    table_ids = [tid for tid, c in counts.items() if c >= 2]
    table_ids.sort()
    return rep_to_emb, rep_to_row, table_ids, D


def table_centroids(z: torch.Tensor, table_ids: torch.Tensor) -> torch.Tensor:
    # returns (B_unique, D)
    unique = torch.unique(table_ids)
    cents = []
    for tid in unique:
        idx = (table_ids == tid).nonzero(as_tuple=True)[0]
        cents.append(z.index_select(0, idx).mean(dim=0))
    return torch.stack(cents, dim=0)





def train(
    cache_root: str,
    model_name: str,
    datasets: List[str],          # <-- ["NQ","WTQ","WIKISQL"]
    reps: List[str],
    out_path: str,
    cfg: TrainCfg,
):
    # Load per-dataset caches and build one big sample list
    dataset_to_rep_to_emb: Dict[str, Dict[str, np.memmap]] = {}
    dataset_to_rep_to_row: Dict[str, Dict[str, Dict[str, int]]] = {}
    samples: List[Tuple[str, str]] = []

    D = None
    for dsname in datasets:
        rep_to_emb, rep_to_row, table_ids, D_ds = load_rep_caches_aligned(cache_root, model_name, dsname, reps)

        if D is None:
            D = D_ds
        elif D_ds != D:
            raise ValueError(f"Embedding dim mismatch across datasets: {dsname} has {D_ds}, expected {D}")

        dataset_to_rep_to_emb[dsname] = rep_to_emb
        dataset_to_rep_to_row[dsname] = rep_to_row
        samples.extend([(dsname, tid) for tid in table_ids])
        print_cache_norm_stats(dataset_to_rep_to_emb[dsname])

    if D is None or len(samples) == 0:
        raise ValueError("No training samples found across datasets (need >=2 reps per table).")

    ds = CacheMultiViewDatasetGiant(
        dataset_to_rep_to_emb=dataset_to_rep_to_emb,
        dataset_to_rep_to_row=dataset_to_rep_to_row,
        samples=samples,
        rep_names=reps,
        max_views=cfg.max_views
    )
    print("Dimension: ",D)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=True,
    )
    #print("D: ",D," | cfg.hidden_mult: ",cfg.hidden_mult," | cfg.alpha: ", cfg.alpha," | cfg.dropout: ", cfg.dropout)
    #adapter = ResidualCentroidAdapter(D, hidden_mult=cfg.hidden_mult, alpha=cfg.alpha, dropout=cfg.dropout).to(cfg.device)
    adapter = BottleneckResidualAdapter(D, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout, use_bias=cfg.use_bias).to(cfg.device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(adapter)
    adapter.train()
    it = iter(dl)

    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    traj_path_jsonl = os.path.join(out_dir, "trajectory.jsonl")
    traj_path_pkl   = os.path.join(out_dir, "trajectory.pkl")

    trajectory = []
    
    meta = {
        "type": "meta",
        "model_name": model_name,
        "datasets": datasets,
        "reps": reps,
        "D": D,
        "cfg": cfg.__dict__,
    }
    with open(traj_path_jsonl, "a") as f:
        f.write(json.dumps(meta) + "\n")

    for step in range(1, cfg.steps + 1):
        try:
            table_ids_views, e_views = next(it)
        except StopIteration:
            it = iter(dl)
            table_ids_views, e_views = next(it)

        table_ids_views = table_ids_views.to(cfg.device)
        e_views = e_views.to(cfg.device)
        z_views = adapter(e_views)
        z_views = F.normalize(z_views, p=2, dim=-1)

        Linv = invariance_to_centroid_loss(z_views, table_ids_views)

        z_tab = table_centroids(z_views, table_ids_views)   # <-- key change
        Lvar = variance_loss(z_views, gamma=cfg.gamma_std)
        Lcov = covariance_loss(z_views)
        #with torch.no_grad():
        #    n_unique = torch.unique(table_ids_views).numel()
        #print("z_tab size",z_tab.shape[0])
        #print("unique_tables", n_unique) 
        e_views_n = F.normalize(e_views, p=2, dim=-1)
        Lid  = identity_preservation_loss(z_views, e_views_n, mode=cfg.id_mode)


        loss = cfg.lam_inv * Linv + cfg.lam_var * Lvar + cfg.lam_cov * Lcov + cfg.lam_id * Lid

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
        opt.step()

        if step % cfg.log_every == 0:
            with torch.no_grad():
                cos_ze = F.cosine_similarity(z_views, e_views, dim=-1).mean().item()
                disp = (z_views - e_views_n).norm(dim=-1).mean().item()
                mean_std = torch.sqrt(z_views.var(dim=0, unbiased=False) + 1e-4).mean().item()
                mean_std_tab = torch.sqrt(z_tab.var(dim=0, unbiased=False) + 1e-4).mean().item()
                mean_norm_e = e_views.norm(dim=-1).mean().item()
                mean_norm_z = z_views.norm(dim=-1).mean().item()
                n_tables = torch.unique(table_ids_views).numel()

            rec = {
                "type": "log",
                "step": step,
                "loss": float(loss.item()),
                "Linv": float(Linv.item()),
                "Lvar": float(Lvar.item()),
                "Lcov": float(Lcov.item()),
                "Lid":  float(Lid.item()),
                "cos_ze": float(cos_ze),
                "mean_disp": float(disp),
                "meanStd_z": float(mean_std),
                "meanStd_z_tab": float(mean_std_tab),
                "meanNorm_e": float(mean_norm_e),
                "meanNorm_z": float(mean_norm_z),
                "unique_tables_in_batch": int(n_tables),
            }
            trajectory.append(rec)

            # append to jsonl in the adapter folder
            with open(traj_path_jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")

            print(
                f"[{model_name}] step={step} loss={rec['loss']:.4f} "
                f"Linv={rec['Linv']:.4f} Lvar={rec['Lvar']:.4f} Lcov={rec['Lcov']:.4f} Lid={rec['Lid']:.4f} "
                f"cos(z,e)={rec['cos_ze']:.3f} mean||z-norm(e)||={rec['mean_disp']:.3f} "
                f"meanStd(z)={rec['meanStd_z']:.3f} meanStd(z_tab)={rec['meanStd_z_tab']:.3f} "
                f"mean||e||={rec['meanNorm_e']:.3f} mean||z||={rec['meanNorm_z']:.3f} "
                f"unique_tables={rec['unique_tables_in_batch']}"
            )


        if (step % cfg.ckpt_every == 0):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(
                {
                    "state_dict": adapter.state_dict(),
                    "D": D,
                    "model_name": model_name,
                    "datasets": datasets,
                    "reps": reps,
                    "cfg": cfg.__dict__,
                },
                out_path,
            )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "D": D,
            "model_name": model_name,
            "datasets": datasets,
            "reps": reps,
            "cfg": cfg.__dict__,
        },
        out_path,
    )
    print(f"Saved adapter to: {out_path}")

    # Save full trajectory as pickle (convenient for plotting)
    with open(traj_path_pkl, "wb") as f:
        pickle.dump(
            {
                "meta": meta,
                "trajectory": trajectory,
            },
            f,
        )
    print(f"Saved training trajectory to:\n  {traj_path_jsonl}\n  {traj_path_pkl}")


@torch.no_grad()
def centroidize_embedding(adapter: ResidualCentroidAdapter, e: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    if device is None:
        device = next(adapter.parameters()).device
    adapter.eval()
    t = torch.from_numpy(np.asarray(e, dtype=np.float32)).to(device)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    z = adapter(t).cpu().numpy().astype(np.float32, copy=False)
    return z[0] if z.shape[0] == 1 else z


if __name__ == "__main__":
    cache_root = "/data/Kushal/UniversalRetrieval_data/emb_cache"
    model_name =  sys.argv[1]
    date = sys.argv[2]
    subset  = sys.argv[3]


    cfg = TrainCfg(
        steps=20000,
        batch_size=512,
        gamma_std=0.05,
        lam_inv=100.0,
        lam_var=25.0,
        lam_cov=1.0,
        lam_id=100.0,
        alpha=0.01,
        r=512,
        max_views=len(REPRESENTATIVE_ORDER)
    )
    if subset == "subset":
        datasets = ["WTQ", "WIKISQL"]  # <-- GIANT corpus "NQ", 
        out_path = f"/data/Kushal/UniversalRetrieval_data/{date}/centroid_adapter_subset_dataset/{model_name}/adapter.pt"#

    else:
        
        datasets = ["NQ", "WTQ", "WIKISQL"]  # <-- GIANT corpus  
        out_path = f"/data/Kushal/UniversalRetrieval_data/{date}/centroid_adapter/{model_name}/adapter.pt"#
    print("Datasets:",datasets)
    train(
        cache_root=cache_root,
        model_name=model_name,
        datasets=datasets,
        reps=REPRESENTATIVE_ORDER,
        out_path=out_path,
        cfg=cfg,
    )
    #  CUDA_VISIBLE_DEVICES=1 python train_universal_adapter.py reasonir > result_reasonir.txt 2>&1
    # CUDA_VISIBLE_DEVICES=1 python test_universal_adapter.py  bge > result_bge.txt 2>&1

    #  CUDA_VISIBLE_DEVICES=1 python train_universal_adapter.py mpnet > result_mpnet.txt 2>&1