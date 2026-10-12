import os
import json
import gc
import numpy as np


def _safe_name(s: str) -> str:
    return s.replace("/", "_")


# ── Dense cache (all models except SPLADE) ────────────────────────────────────

def cache_paths(cache_root, model_name, dataset, rep_name):
    rep_name = _safe_name(rep_name)
    rep_dir = os.path.join(cache_root, model_name, dataset)
    os.makedirs(rep_dir, exist_ok=True)
    emb_path = os.path.join(rep_dir, f"{rep_name}.emb.npy")
    ids_path = os.path.join(rep_dir, f"{rep_name}.doc_ids.json")
    return emb_path, ids_path

def cache_exists(cache_root, model_name, dataset, rep_name):
    emb_path, ids_path = cache_paths(cache_root, model_name, dataset, rep_name)
    return os.path.exists(emb_path) and os.path.exists(ids_path)

def save_cache(cache_root, model_name, dataset, rep_name, doc_ids, emb_np_float32):
    emb_path, ids_path = cache_paths(cache_root, model_name, dataset, rep_name)
    np.save(emb_path, emb_np_float32.astype(np.float32, copy=False))
    with open(ids_path, "w") as f:
        json.dump([str(x) for x in doc_ids], f)

def load_cache_mmap(cache_root, model_name, dataset, rep_name):
    emb_path, ids_path = cache_paths(cache_root, model_name, dataset, rep_name)
    with open(ids_path, "r") as f:
        doc_ids = json.load(f)
    emb = np.load(emb_path, mmap_mode="r")  # (N, D) memory-mapped
    return doc_ids, emb


# ── Sparse cache (SPLADE) ─────────────────────────────────────────────────────

def sparse_cache_paths(cache_root, model_name, dataset, rep_name):
    rep_name = _safe_name(rep_name)
    rep_dir = os.path.join(cache_root, model_name, dataset)
    os.makedirs(rep_dir, exist_ok=True)
    emb_path = os.path.join(rep_dir, f"{rep_name}.sparse.npz")
    ids_path = os.path.join(rep_dir, f"{rep_name}.doc_ids.json")
    return emb_path, ids_path

def sparse_cache_exists(cache_root, model_name, dataset, rep_name):
    emb_path, ids_path = sparse_cache_paths(cache_root, model_name, dataset, rep_name)
    return os.path.exists(emb_path) and os.path.exists(ids_path)

def save_sparse_cache(cache_root, model_name, dataset, rep_name, doc_ids, sparse_mat):
    """Save a scipy CSR sparse matrix as .sparse.npz (space-efficient)."""
    import scipy.sparse
    emb_path, ids_path = sparse_cache_paths(cache_root, model_name, dataset, rep_name)
    scipy.sparse.save_npz(emb_path, sparse_mat.asformat("csr"))
    with open(ids_path, "w") as f:
        json.dump([str(x) for x in doc_ids], f)

def load_sparse_cache(cache_root, model_name, dataset, rep_name):
    """Load a scipy CSR sparse matrix from .sparse.npz."""
    import scipy.sparse
    emb_path, ids_path = sparse_cache_paths(cache_root, model_name, dataset, rep_name)
    with open(ids_path) as f:
        doc_ids = json.load(f)
    sparse_mat = scipy.sparse.load_npz(emb_path).asformat("csr")
    return doc_ids, sparse_mat


# ── Model-aware dispatch ──────────────────────────────────────────────────────

def cache_exists_for_model(cache_root, model_name, dataset, rep_name):
    """Check cache existence, using sparse format for SPLADE."""
    if model_name == "splade":
        return sparse_cache_exists(cache_root, model_name, dataset, rep_name)
    return cache_exists(cache_root, model_name, dataset, rep_name)

def load_cache_for_model(cache_root, model_name, dataset, rep_name):
    """Load cache; returns scipy CSR for SPLADE, numpy memmap for others."""
    if model_name == "splade":
        return load_sparse_cache(cache_root, model_name, dataset, rep_name)
    return load_cache_mmap(cache_root, model_name, dataset, rep_name)
