import os, json, gc
import numpy as np
import torch

def _safe_name(s: str) -> str:
    return s.replace("/", "_")

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