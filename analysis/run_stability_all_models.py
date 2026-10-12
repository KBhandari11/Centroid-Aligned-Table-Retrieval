"""
Empirical Stability Certificate Analysis — all models
======================================================
Computes Delta, rho, R = rho/Delta for every (model, dataset, query, format, K)
using pre-cached table embeddings and pre-trained adapter checkpoints.

Usage
-----
  # Full run (all models, all queries):
  python -m analysis.run_stability_all_models

  # Quick verification (100 queries per dataset):
  python -m analysis.run_stability_all_models --max_queries 100

  # Single model:
  python -m analysis.run_stability_all_models --models mpnet bge

  # Single dataset:
  python -m analysis.run_stability_all_models --datasets WTQ

Output
------
  data/stability_analysis_all/{model}/stability_metrics.csv
  data/stability_analysis_all/{model}/prop3_drel.csv
  data/stability_analysis_all/{model}/table_A_margin.csv
  data/stability_analysis_all/{model}/table_B_cert_rates.csv
  data/stability_analysis_all/{model}/table_C_adapter_effect.csv
  data/stability_analysis_all/{model}/table_D_prop3.csv
  data/stability_analysis_all/{model}/fig_*.{pdf,png}
  data/stability_analysis_all/summary_all_models.csv   ← cross-model summary
"""

import argparse
import gc
import json
import logging
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script runs
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Repo root on path ─────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from utils.cache_embedding import cache_exists_for_model, load_cache_for_model
from utils.core import CACHE_ROOT, FORMATS_17, SPLADE_TOPK, _topk_dense_to_sparse
from utils.Adapter import HierDenseAdapter, HierSparseAdapter, FMT2IDX

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================
DATA_ROOT      = '/data/Kushal/UniversalRetrieval_data'
QUESTIONS_PATH = f'{DATA_ROOT}/dataset_all/new_all_questions_dict_wtq_nqt_wikisql.pkl'
RESULTS_ROOT   = os.path.join(REPO_ROOT, 'data', 'stability_analysis_all')

DELTA_ALL = 0.01       # Proposition 3 premise (HierCfg.delta_all)
ALL_DATASETS = ['WTQ', 'WIKISQL', 'NQ']
K_VALUES     = [1, 5, 10]

# Models with available embeddings + adapter paths (prefer subset adapter, fall back to full)
MODEL_CONFIGS = {
    'mpnet':    {'is_sparse': False},
    'miniLM':   {'is_sparse': False},
    'mdbr':     {'is_sparse': False},
    'bge':      {'is_sparse': False},
    'qwen0.6B': {'is_sparse': False},
    'qwen4B':   {'is_sparse': False},
    'reasonir': {'is_sparse': False},
    'splade':   {'is_sparse': True},
}

def _adapter_path(model_name):
    """Return the best available adapter path for a model, or None."""
    candidates = [
        f'{DATA_ROOT}/2026-09-12/centroid_adapter_subset_dataset/{model_name}/adapter.pt',
        f'{DATA_ROOT}/2026-09-17/centroid_adapter/{model_name}/adapter.pt',
        f'{DATA_ROOT}/2026-09-12/centroid_adapter/{model_name}/adapter.pt',
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# =============================================================================
# Core helpers
# =============================================================================

def group_chunks_by_table(chunk_ids):
    tbl2idx = defaultdict(list)
    for i, did in enumerate(chunk_ids):
        tbl2idx[did.rsplit('::', 1)[0]].append(i)
    table_ids = sorted(tbl2idx.keys())
    return tbl2idx, table_ids


def compute_table_scores(chunk_ids, chunk_emb, Q_matrix, q_batch=128, c_batch=4096):
    """
    Returns (table_ids, score_matrix) where score_matrix is (n_tables, n_queries).
    Table score = top-3-chunk-mean (matching the retrieval pipeline's primary score).
    """
    is_sp    = scipy.sparse.issparse(chunk_emb)
    n_chunks = chunk_emb.shape[0]
    n_queries = Q_matrix.shape[0]

    tbl2idx, table_ids = group_chunks_by_table(chunk_ids)
    n_tables = len(table_ids)
    scores   = np.zeros((n_tables, n_queries), dtype=np.float32)

    for q0 in range(0, n_queries, q_batch):
        q1  = min(q0 + q_batch, n_queries)
        Qb  = np.ascontiguousarray(Q_matrix[q0:q1], dtype=np.float32)

        if is_sp:
            cs = np.asarray(chunk_emb.dot(Qb.T), dtype=np.float32)  # (n_c, bq)
        else:
            bq = q1 - q0
            cs = np.zeros((n_chunks, bq), dtype=np.float32)
            for c0 in range(0, n_chunks, c_batch):
                c1 = min(c0 + c_batch, n_chunks)
                cs[c0:c1] = np.asarray(chunk_emb[c0:c1], dtype=np.float32) @ Qb.T

        for t_idx, tid in enumerate(table_ids):
            rows = tbl2idx[tid]
            seg  = cs[rows]  # (n_c_t, bq)
            nc   = seg.shape[0]
            agg  = seg.mean(0) if nc <= 3 else np.partition(seg, -3, axis=0)[-3:].mean(0)
            scores[t_idx, q0:q1] = agg

    return table_ids, scores


def rel_sq_dist(a, b, eps=1e-8):
    return float(((a - b) ** 2).sum() / ((a ** 2).sum() + eps))


def stability_metrics(cen_scores, fmt_scores, table_ids, qids, questions_dict,
                      K_values, source_tag, fmt_name):
    """
    Returns list of row-dicts for every (query, K).
    cen_scores, fmt_scores: (n_tables, n_queries) aligned to the same table order.
    """
    tid2idx = {t: i for i, t in enumerate(table_ids)}
    n_tables = len(table_ids)
    rows = []

    for q_idx, qid in enumerate(qids):
        gold = str(questions_dict[qid]['gold_table_id'])
        if gold not in tid2idx:
            continue
        g = tid2idx[gold]

        cen_q = cen_scores[:, q_idx]
        fmt_q = fmt_scores[:, q_idx]

        mask        = np.ones(n_tables, dtype=bool); mask[g] = False
        cen_irr     = cen_q[mask];  fmt_irr  = fmt_q[mask]
        res_irr     = fmt_irr - cen_irr
        res_gold    = fmt_q[g] - cen_q[g]

        order           = np.argsort(cen_irr)[::-1]
        cen_irr_sorted  = cen_irr[order]

        rho = float(res_irr.max() - res_gold)

        for K in K_values:
            kb    = min(K - 1, len(cen_irr_sorted) - 1)
            Delta = float(cen_q[g] - cen_irr_sorted[kb])
            R     = float(rho / Delta) if Delta > 0 else float('nan')
            rows.append(dict(
                qid=qid, fmt=fmt_name, K=K,
                Delta=Delta, rho=rho, R=R,
                certified=bool(Delta > 0 and rho < Delta),
                gold_rank_centroid=int((cen_irr > cen_q[g]).sum()) + 1,
                source=source_tag,
            ))
    return rows


# =============================================================================
# Query encoding + caching
# =============================================================================

def encode_queries(model_name, questions_dict, cache_dir, batch_size=256):
    cache_emb = os.path.join(cache_dir, 'query_emb.npy')
    cache_ids = os.path.join(cache_dir, 'query_ids.json')

    if os.path.exists(cache_emb) and os.path.exists(cache_ids):
        qids = json.load(open(cache_ids))
        Q    = np.load(cache_emb, mmap_mode='r')
        log.info(f'  Query cache hit: {len(qids)} queries, shape {Q.shape}')
        return qids, np.asarray(Q, dtype=np.float32)

    log.info('  Encoding queries (no cache)...')
    from utils.model import get_model
    from accelerate import Accelerator
    acc   = Accelerator()
    model = get_model(acc, model_name=model_name)

    qids  = sorted(questions_dict.keys())
    texts = [str(questions_dict[q]['question']) for q in qids]
    embs  = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f'Encoding {model_name} queries'):
        batch = texts[i:i+batch_size]
        if model_name == 'reasonir':
            q = model.encode(batch, instruction='')
        elif model_name == 'jina':
            q = model.encode(batch, task='retrieval.query', prompt_name='retrieval.query')
        elif model_name == 'splade':
            q = model.encode_query(batch).to_dense()
        else:
            q = model.encode(batch)
        if isinstance(q, torch.Tensor):
            q = q.detach().cpu().float().numpy()
        embs.append(np.asarray(q, dtype=np.float32))

    Q = np.concatenate(embs, axis=0).astype(np.float32)
    np.save(cache_emb, Q)
    json.dump(qids, open(cache_ids, 'w'))
    log.info(f'  Encoded and cached {len(qids)} queries, shape {Q.shape}')

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return qids, Q


# =============================================================================
# Adapter
# =============================================================================

def load_adapter(path, device):
    ckpt    = torch.load(path, map_location='cpu', weights_only=False)
    D       = int(ckpt['D'])
    cfg     = ckpt.get('cfg', {})
    is_sp   = (ckpt.get('model_name') == 'splade')
    if is_sp:
        adapter = HierSparseAdapter(D, r=cfg.get('r', 256), dropout=0.0,
                                    k_exp=cfg.get('k_exp', 0), topk=SPLADE_TOPK)
    else:
        adapter = HierDenseAdapter(D, r=cfg.get('r', 256), dropout=0.0)
    adapter.load_state_dict(ckpt['state_dict'], strict=True)
    adapter.eval().to(device)
    return adapter, D, is_sp


def apply_adapter(adapter, chunk_emb, fmt_key, is_sp, device, batch_size=8192):
    base   = fmt_key.removeprefix('chunked_')
    fmt_id = FMT2IDX.get(base, None)
    N      = chunk_emb.shape[0]
    outs   = [] if is_sp else np.zeros((N, chunk_emb.shape[1]), dtype=np.float32)

    with torch.no_grad():
        for s in range(0, N, batch_size):
            e   = min(s + batch_size, N)
            blk = chunk_emb[s:e]
            blk = blk.toarray() if scipy.sparse.issparse(blk) else np.asarray(blk)
            t   = torch.from_numpy(np.ascontiguousarray(blk, dtype=np.float32)).to(device)
            fmt = (torch.full((t.shape[0],), fmt_id, dtype=torch.long, device=device)
                   if fmt_id is not None else None)
            z   = adapter(t, fmt)[0].detach().cpu().numpy().astype(np.float32, copy=False)
            if is_sp:
                outs.append(_topk_dense_to_sparse(z, SPLADE_TOPK))
            else:
                outs[s:e] = z

    return scipy.sparse.vstack(outs, format='csr') if is_sp else outs


# =============================================================================
# Reporting helpers
# =============================================================================

def make_tables(df_stab, df_prop3, query_data, out_dir):
    df_base = df_stab[df_stab.source == 'base'].copy()

    # Table A: margin distribution
    rows = []
    for (ds, K), g in df_base.groupby(['dataset', 'K']):
        dq = g.groupby('qid')['Delta'].first()
        rows.append(dict(Dataset=ds, K=K, N=len(dq),
                         Frac_Delta_pos=round((dq > 0).mean(), 3),
                         Median_Delta=round(dq.median(), 4),
                         P25_Delta=round(dq.quantile(0.25), 4),
                         P75_Delta=round(dq.quantile(0.75), 4)))
    pd.DataFrame(rows).to_csv(f'{out_dir}/table_A_margin.csv', index=False)

    # Table B: base certification rates
    df_cert = df_base[df_base.Delta > 0]
    rows = []
    for (ds, K), g in df_cert.groupby(['dataset', 'K']):
        cfmt = g.groupby('fmt')['certified'].mean()
        rows.append(dict(Dataset=ds, K=K,
                         N_certifiable=g.groupby('qid').ngroups,
                         Mean_R=round(g['R'].mean(), 3),
                         Median_R=round(g['R'].median(), 3),
                         Frac_cert_mean=round(cfmt.mean(), 3),
                         Frac_cert_worst=round(cfmt.min(), 3),
                         Worst_fmt=cfmt.idxmin(),
                         Frac_cert_best=round(cfmt.max(), 3),
                         Best_fmt=cfmt.idxmax()))
    pd.DataFrame(rows).to_csv(f'{out_dir}/table_B_cert_rates.csv', index=False)

    # Table C: adapter effect
    if 'adapted' in df_stab.source.values:
        df_ad = df_stab[(df_stab.source == 'adapted') & (df_stab.Delta > 0)]
        rows = []
        for (ds, K), b in df_cert.groupby(['dataset', 'K']):
            a = df_ad[(df_ad.dataset == ds) & (df_ad.K == K)]
            if a.empty: continue
            rows.append(dict(Dataset=ds, K=K,
                             Cert_base=round(b['certified'].mean(), 3),
                             Cert_adapted=round(a['certified'].mean(), 3),
                             Mean_R_base=round(b['R'].mean(), 3),
                             Mean_R_adapted=round(a['R'].mean(), 3),
                             Median_R_base=round(b['R'].median(), 3),
                             Median_R_adapted=round(a['R'].median(), 3)))
        pd.DataFrame(rows).to_csv(f'{out_dir}/table_C_adapter_effect.csv', index=False)

    # Table D: Proposition 3 bound
    if not df_prop3.empty and 'adapted' in df_stab.source.values:
        # Precompute per-dataset q norms and max centroid norms
        ds_meta = {}
        for ds in df_prop3['dataset'].unique():
            qids_ds, Q_ds = query_data[ds]
            q_norms   = np.linalg.norm(Q_ds, axis=1)
            # centroid norms were not stored; approximate from base Delta stats —
            # use the cached centroid embedding
            centroid_key = 'chunked_centroid_all'
            model_name_local = df_stab['model'].iloc[0]
            max_c_norm = float('nan')
            if cache_exists_for_model(CACHE_ROOT, model_name_local, ds, centroid_key):
                c_ids, c_emb = load_cache_for_model(CACHE_ROOT, model_name_local, ds, centroid_key)
                tbl2idx_c, tids_c = group_chunks_by_table(c_ids)
                c_norms = [
                    np.linalg.norm(np.asarray(c_emb[tbl2idx_c[tid]], dtype=np.float32).mean(0))
                    for tid in tids_c
                ]
                max_c_norm = float(np.max(c_norms))
                del c_emb; gc.collect()
            ds_meta[ds] = dict(q_norms=q_norms, max_c_norm=max_c_norm,
                               qid_to_idx={q: i for i, q in enumerate(qids_ds)})

        rows = []
        for (ds, fmt), g in df_prop3.groupby(['dataset', 'fmt']):
            meta  = ds_meta[ds]
            bound = 0.2 * meta['q_norms'] * meta['max_c_norm']
            sub   = df_base[(df_base.dataset == ds) & (df_base.fmt == fmt) & (df_base.K == 1)]
            qd    = sub.set_index('qid')['Delta']
            checks = [bound[meta['qid_to_idx'][q]] < d
                      for q, d in qd.items() if d > 0 and q in meta['qid_to_idx']]
            rows.append(dict(Dataset=ds, Format=fmt,
                             Frac_premise=round(g['premise_holds'].mean(), 3),
                             Max_c_norm=round(meta['max_c_norm'], 3),
                             Mean_bound=round(bound.mean(), 3),
                             Frac_certifies=round(float(np.mean(checks)) if checks else 0.0, 3)))
        pd.DataFrame(rows).to_csv(f'{out_dir}/table_D_prop3.csv', index=False)


def make_figures(df_stab, model_name, out_dir, data_root):
    df_base = df_stab[df_stab.source == 'base']
    df_cert = df_base[df_base.Delta > 0]
    datasets = sorted(df_cert['dataset'].unique())
    if not datasets:
        return

    # ── Fig 1: cert rate vs K ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=False)
    if len(datasets) == 1: axes = [axes]
    for ax, ds in zip(axes, datasets):
        g  = df_cert[df_cert.dataset == ds]
        ks = sorted(g['K'].unique())
        ax.plot(ks, [g[g.K==k]['certified'].mean() for k in ks],
                marker='o', lw=2, label='Mean')
        worst = [g[g.K==k].groupby('fmt')['certified'].mean().min() for k in ks]
        best  = [g[g.K==k].groupby('fmt')['certified'].mean().max() for k in ks]
        ax.fill_between(ks, worst, best, alpha=0.2, label='Format range')
        ax.axhline(1.0, ls='--', lw=0.8, color='gray')
        ax.set(xlabel='K', ylabel='Frac certified (R<1)', title=ds, xticks=ks, ylim=(0, 1.05))
        ax.legend(fontsize=8)
    fig.suptitle(f'Cert Rate vs K  [{model_name}]', fontsize=13)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_dir}/fig_depth_monotonicity.{ext}',
                    bbox_inches='tight', dpi=150 if ext == 'png' else None)
    plt.close(fig)

    # ── Fig 2: R-value CDF with failure overlay ──────────────────────────────
    k_vals = [k for k in [1, 10] if k in df_cert['K'].unique()]
    if not k_vals: k_vals = sorted(df_cert['K'].unique())[:2]
    fig, axes = plt.subplots(len(datasets), len(k_vals),
                             figsize=(5 * len(k_vals), 4 * len(datasets)), squeeze=False)
    for ri, ds in enumerate(datasets):
        for ci, K in enumerate(k_vals):
            ax = axes[ri][ci]
            g  = df_cert[(df_cert.dataset == ds) & (df_cert.K == K)]
            if g.empty: ax.set_title(f'{ds} K={K}'); continue
            R_s = np.sort(g['R'].dropna().values)
            ax.plot(R_s, np.arange(1, len(R_s)+1)/len(R_s), lw=1.5, label='CDF(R)')
            ax.axvline(1.0, ls='--', lw=1.2, color='red', label='R=1')
            # overlay failures from pre-computed rank CSVs
            rank_root = f'{data_root}/retrieval_all/{model_name}_results_rank'
            for fmt in g['fmt'].unique()[:3]:
                rpath = f'{rank_root}/{ds}/chunked_{fmt}_gold_rank_per_question.csv'
                if not os.path.exists(rpath): continue
                rdf   = pd.read_csv(rpath)
                ranks = dict(zip(rdf['question_id'].astype(str), rdf['rank']))
                sub   = g[g.fmt == fmt].dropna(subset=['R'])
                fail  = sub[sub['qid'].map(lambda q: ranks.get(str(q), 9999)) > K]['R'].values
                if len(fail):
                    ax.scatter(fail, np.ones(len(fail))*1.02, marker='x', s=12,
                               color='red', alpha=0.3)
            ax.set(xlabel='R', ylabel='CDF', title=f'{ds} K={K}')
            ax.legend(fontsize=7)
    fig.suptitle(f'R Distribution  [{model_name}]', fontsize=12)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_dir}/fig_R_distribution.{ext}',
                    bbox_inches='tight', dpi=150 if ext == 'png' else None)
    plt.close(fig)

    # ── Fig 3: Base vs Adapted R CDF (K=1) ───────────────────────────────────
    if 'adapted' in df_stab.source.values:
        df_ad = df_stab[(df_stab.source == 'adapted') & (df_stab.Delta > 0)]
        fig, axes = plt.subplots(1, len(datasets),
                                 figsize=(5 * len(datasets), 4), squeeze=False)
        axes = axes[0]
        for ax, ds in zip(axes, datasets):
            K  = 1
            b  = df_cert[(df_cert.dataset==ds) & (df_cert.K==K)]['R'].dropna().values
            a  = df_ad[(df_ad.dataset==ds) & (df_ad.K==K)]['R'].dropna().values
            for vals, label, color in [(b, 'Base', 'steelblue'), (a, 'Adapted', 'darkorange')]:
                if not len(vals): continue
                s = np.sort(vals)
                ax.plot(s, np.arange(1, len(s)+1)/len(s), lw=2, label=label, color=color)
            ax.axvline(1.0, ls='--', lw=1.2, color='red')
            ax.set(xlabel='R', ylabel='CDF', title=f'{ds} K={K}')
            ax.legend(fontsize=9)
        fig.suptitle(f'Base vs Adapted R  [{model_name}]', fontsize=12)
        plt.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(f'{out_dir}/fig_R_base_vs_adapted.{ext}',
                        bbox_inches='tight', dpi=150 if ext == 'png' else None)
        plt.close(fig)

    # ── Fig 4: per-format cert heatmap ────────────────────────────────────────
    for ds in datasets:
        sub = df_cert[df_cert.dataset == ds]
        if sub.empty: continue
        pivot = sub.groupby(['fmt', 'K'])['certified'].mean().unstack('K')
        pivot = pivot.reindex(FORMATS_17)
        fig, ax = plt.subplots(figsize=(4 + len(K_VALUES), 7))
        im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'K={k}' for k in pivot.columns], fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(f'Cert Rate by Format — {ds}  [{model_name}]')
        plt.colorbar(im, ax=ax, label='Frac certified (R<1)')
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                            color='white' if v < 0.4 or v > 0.8 else 'black')
        plt.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(f'{out_dir}/fig_cert_heatmap_{ds}.{ext}',
                        bbox_inches='tight', dpi=150 if ext == 'png' else None)
        plt.close(fig)


# =============================================================================
# Per-model runner
# =============================================================================

def run_model(model_name, datasets, all_questions, max_queries, device):
    log.info(f'{"="*60}')
    log.info(f'MODEL: {model_name}')
    log.info(f'{"="*60}')

    out_dir    = os.path.join(RESULTS_ROOT, model_name)
    os.makedirs(out_dir, exist_ok=True)

    is_sparse  = MODEL_CONFIGS[model_name]['is_sparse']
    adapter_path = _adapter_path(model_name)
    if adapter_path:
        log.info(f'  Adapter: {adapter_path}')
    else:
        log.info(f'  No adapter found — skipping adapted metrics')

    # Load adapter
    adapter = adapter_D = None
    if adapter_path:
        adapter, adapter_D, _ = load_adapter(adapter_path, device)
        log.info(f'  Adapter loaded: D={adapter_D}')

    # Encode / load queries per dataset
    query_data = {}
    q_cache_root = os.path.join(DATA_ROOT, 'query_emb_cache', model_name)
    for ds in datasets:
        ds_cache = os.path.join(q_cache_root, ds)
        os.makedirs(ds_cache, exist_ok=True)
        qids, Q = encode_queries(model_name, all_questions[ds], ds_cache)
        if max_queries is not None and len(qids) > max_queries:
            qids = qids[:max_queries]
            Q    = Q[:max_queries]
            log.info(f'  [{ds}] capped to {max_queries} queries')
        query_data[ds] = (qids, Q)

    all_rows   = []
    prop3_rows = []

    for dataset in datasets:
        log.info(f'\n  ── Dataset: {dataset} ──')
        qids, Q = query_data[dataset]

        centroid_key = 'chunked_centroid_all'
        if not cache_exists_for_model(CACHE_ROOT, model_name, dataset, centroid_key):
            log.warning(f'  No centroid_all cache for {model_name}/{dataset}, skipping.')
            continue

        cen_ids, cen_emb = load_cache_for_model(CACHE_ROOT, model_name, dataset, centroid_key)
        log.info(f'  centroid_all chunks: {len(cen_ids)}, shape {cen_emb.shape}')

        cen_table_ids, centroid_scores = compute_table_scores(cen_ids, cen_emb, Q)
        log.info(f'  centroid scored: {len(cen_table_ids)} tables × {len(qids)} queries')

        # Pre-aggregate centroid embeddings for Prop 3 d_rel check
        cen_table_emb = None
        if not is_sparse and adapter is not None:
            tbl2idx_cen, _ = group_chunks_by_table(cen_ids)
            cen_emb_arr    = np.asarray(cen_emb, dtype=np.float32)
            cen_table_emb  = {
                tid: cen_emb_arr[idxs].mean(0)
                for tid, idxs in tbl2idx_cen.items()
            }

        del cen_emb; gc.collect()

        for fmt in tqdm(FORMATS_17, desc=f'  {dataset} formats'):
            fmt_key = f'chunked_{fmt}'
            if not cache_exists_for_model(CACHE_ROOT, model_name, dataset, fmt_key):
                continue

            fmt_ids, fmt_emb = load_cache_for_model(CACHE_ROOT, model_name, dataset, fmt_key)
            fmt_table_ids, fmt_scores_base = compute_table_scores(fmt_ids, fmt_emb, Q)

            # Align to common tables
            cen_idx_map = {t: i for i, t in enumerate(cen_table_ids)}
            fmt_idx_map = {t: i for i, t in enumerate(fmt_table_ids)}
            common      = sorted(set(fmt_table_ids) & set(cen_table_ids))
            cen_ord     = np.array([cen_idx_map[t] for t in common])
            fmt_ord     = np.array([fmt_idx_map[t] for t in common])

            cen_aln = centroid_scores[cen_ord]
            fmt_aln = fmt_scores_base[fmt_ord]

            rows = stability_metrics(cen_aln, fmt_aln, common, qids,
                                     all_questions[dataset], K_VALUES, 'base', fmt)
            for r in rows:
                r['dataset'] = dataset; r['model'] = model_name
            all_rows.extend(rows)

            # Adapted
            if adapter is not None and not is_sparse:
                fmt_emb_ad = apply_adapter(adapter, fmt_emb, fmt_key, is_sparse, device)
                _, fmt_scores_ad = compute_table_scores(fmt_ids, fmt_emb_ad, Q)
                fmt_aln_ad = fmt_scores_ad[fmt_ord]

                rows_ad = stability_metrics(cen_aln, fmt_aln_ad, common, qids,
                                            all_questions[dataset], K_VALUES, 'adapted', fmt)
                for r in rows_ad:
                    r['dataset'] = dataset; r['model'] = model_name
                all_rows.extend(rows_ad)

                # Prop 3 d_rel
                if cen_table_emb is not None:
                    tbl2idx_fmt, _ = group_chunks_by_table(fmt_ids)
                    ad_arr = np.asarray(fmt_emb_ad, dtype=np.float32)
                    for tid in common:
                        if tid not in tbl2idx_fmt or tid not in cen_table_emb:
                            continue
                        idxs    = tbl2idx_fmt[tid]
                        z_tilde = ad_arr[idxs].mean(0)
                        d       = rel_sq_dist(cen_table_emb[tid], z_tilde)
                        prop3_rows.append(dict(
                            model=model_name, dataset=dataset, fmt=fmt, tid=tid,
                            d_rel=d, premise_holds=bool(d <= DELTA_ALL),
                        ))

            del fmt_emb; gc.collect()

        del centroid_scores; gc.collect()
        log.info(f'  Rows so far: {len(all_rows)}')

    # ── Save raw data ─────────────────────────────────────────────────────────
    df_stab  = pd.DataFrame(all_rows)
    df_prop3 = pd.DataFrame(prop3_rows) if prop3_rows else pd.DataFrame()

    df_stab.to_csv(f'{out_dir}/stability_metrics.csv', index=False)
    if not df_prop3.empty:
        df_prop3.to_csv(f'{out_dir}/prop3_drel.csv', index=False)

    log.info(f'  Saved {len(df_stab)} stability rows → {out_dir}/stability_metrics.csv')

    # ── Tables and figures ────────────────────────────────────────────────────
    if not df_stab.empty:
        make_tables(df_stab, df_prop3, query_data, out_dir)
        make_figures(df_stab, model_name, out_dir, DATA_ROOT)
        log.info(f'  Tables and figures written to {out_dir}')

    return df_stab


# =============================================================================
# Cross-model summary
# =============================================================================

def write_summary(all_dfs):
    rows = []
    for df in all_dfs:
        if df.empty: continue
        model = df['model'].iloc[0]
        df_b  = df[(df.source == 'base') & (df.Delta > 0)]
        df_ad = df[(df.source == 'adapted') & (df.Delta > 0)] if 'adapted' in df.source.values else pd.DataFrame()
        for (ds, K), g in df_b.groupby(['dataset', 'K']):
            a = df_ad[(df_ad.dataset == ds) & (df_ad.K == K)] if not df_ad.empty else pd.DataFrame()
            rows.append(dict(
                Model=model, Dataset=ds, K=K,
                N_certifiable=g.groupby('qid').ngroups,
                Frac_cert_base=round(g['certified'].mean(), 3),
                Frac_cert_adapted=round(a['certified'].mean(), 3) if not a.empty else None,
                Median_R_base=round(g['R'].median(), 3),
                Median_R_adapted=round(a['R'].median(), 3) if not a.empty else None,
                Mean_R_base=round(g['R'].mean(), 3),
                Frac_Delta_pos=round((df_b[df_b.dataset==ds][df_b.K==K]
                                      .groupby('qid')['Delta'].first() > 0).mean(), 3),
            ))
    summary = pd.DataFrame(rows)
    path = os.path.join(RESULTS_ROOT, 'summary_all_models.csv')
    summary.to_csv(path, index=False)
    log.info(f'\nCross-model summary → {path}')
    print('\n' + summary.to_string(index=False))
    return summary


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stability certificate analysis — all models')
    parser.add_argument('--models', nargs='+', default=list(MODEL_CONFIGS.keys()),
                        help='Models to run (default: all)')
    parser.add_argument('--datasets', nargs='+', default=ALL_DATASETS,
                        help='Datasets to evaluate (default: WTQ WIKISQL NQ)')
    parser.add_argument('--max_queries', type=int, default=None,
                        help='Cap queries per dataset (None = all). Use 100 for quick check.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    log.info(f'Results root: {RESULTS_ROOT}')
    log.info(f'Models: {args.models}')
    log.info(f'Datasets: {args.datasets}')
    log.info(f'MAX_QUERIES: {args.max_queries}')
    log.info(f'Device: {args.device}')

    with open(QUESTIONS_PATH, 'rb') as f:
        all_questions = pickle.load(f)

    all_dfs = []
    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            log.warning(f'Unknown model "{model_name}", skipping.')
            continue
        try:
            df = run_model(model_name, args.datasets, all_questions,
                           args.max_queries, args.device)
            all_dfs.append(df)
        except Exception as e:
            log.error(f'Model {model_name} failed: {e}', exc_info=True)
        finally:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    if all_dfs:
        write_summary(all_dfs)

    log.info('Done.')


if __name__ == '__main__':
    main()
