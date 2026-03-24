
import pickle
import torch
import heapq
import numpy as np
from .cache_embedding import *

category = {
    "Popular Representation": ["pipe_serialized", "token_serialized", "space_serialized"],
    "Data Representation": ["csv", "tsv", "html", "markdown", "latex", "dict", "json", "xml"],
    "Structural Transformations": ["shuffled_rows", "shuffled_cols", "transpose"],
    "Schema and Definition Types": ["mschema", "macschema", "ddl"],
}
# Names you'll use in retrieval
CENTROID_KEYS = {
    "Popular Representation": "centroid_popular",
    "Data Representation": "centroid_data",
    "Structural Transformations": "centroid_structural",
    "Schema and Definition Types": "centroid_schema",
}
FINAL_CENTROID_KEY = "centroid_all"   # average of the 4 category-centroids
CACHE_ROOT = "./emb_cache" 



def get_data(filename: str)-> list: 
        '''
        Get the data from the jsonl file
        '''
        with open(filename,"rb") as f:
            question_data=pickle.load(f)
        return question_data

def get_table_corpus(save_table_real_data_file_path):
        all_tables = {}
        if os.path.exists(save_table_real_data_file_path):
                with open(save_table_real_data_file_path,"rb") as f:
                        all_tables = pickle.load(f)
        return all_tables 

def get_transform_table_corpus(save_wikitable_tranformed_tables):
        with open(save_wikitable_tranformed_tables,"rb") as f:
                table_corpus=pickle.load(f)
        return table_corpus

def _to_text(x):
    return x if isinstance(x, str) else str(x)

def _l2_normalize_rows_torch(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return X / (X.norm(dim=1, keepdim=True) + eps)

def compute_and_cache_rep_embeddings(
    model_name,
    model,
    table_corpus,           # dict: table_id -> {rep_name: text, ...}
    table_ids,              # list of table_ids to consider
    rep_name,
    cache_root,
    dataset,
    batch_size=256,
    normalize=False,
    instruction="",
):
    docs, doc_ids = [], []
    for tid in table_ids:
        if tid not in table_corpus:
            continue
        if rep_name not in table_corpus[tid]:
            continue
        table = table_corpus[tid][rep_name]
        # Ensure string
        if not isinstance(table, str):
            table = str(table)
        # Avoid truly empty strings
        table = table.strip()
        if table == "":
            table = "[EMPTY]"
        docs.append(table)
        doc_ids.append(tid)

    if len(doc_ids) == 0:
        # cache an empty entry if you want, or just return
        return None, None

    all_emb = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
            
        if model_name == "reasonir":
            emb = model.encode(batch, instruction=instruction)
        elif model_name in ("bge", "mpnet"):
            emb = model.encode(batch)
        elif model_name == "jina":
            emb = model.encode(batch, task="retrieval.passage", prompt_name="retrieval.passage")
        elif model_name == "splade":
            # splade returns sparse-like; you convert to dense
            emb = model.encode_document(batch).to_dense()
        else:
            emb = model.encode(batch)

        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().float().numpy()
        else:
            emb = np.asarray(emb, dtype=np.float32)

        if normalize:
            denom = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb = emb / denom

        all_emb.append(emb)

    emb_np = np.concatenate(all_emb, axis=0).astype(np.float32, copy=False)
    save_cache(cache_root, model_name, dataset, rep_name, doc_ids, emb_np)
    return doc_ids, emb_np

def compute_rep_combination_in_memory(
    cache_root,
    model_name,
    dataset,
    rep_list,
    normalize=False,
):
    """
    Load per-rep cached embeddings and return their mean combination in memory.
    Does NOT save to disk.
    Returns: (doc_ids, combo_emb_np) where combo_emb_np shape is [N, D].
    """
    if len(rep_list) == 0:
        return None, None

    # first available rep as anchor
    anchor = None
    for r in rep_list:
        if cache_exists(cache_root, model_name, dataset, r):
            anchor = r
            break
    if anchor is None:
        return None, None

    doc_ids0, emb0 = load_cache_mmap(cache_root, model_name, dataset, anchor)
    N, D = emb0.shape

    acc = np.zeros((N, D), dtype=np.float32)
    cnt = np.zeros((N,), dtype=np.float32)

    for r in rep_list:
        if not cache_exists(cache_root, model_name, dataset, r):
            continue

        doc_ids_r, emb_r = load_cache_mmap(cache_root, model_name, dataset, r)
        if doc_ids_r != doc_ids0:
            raise ValueError(
                f"doc_ids mismatch between {anchor} and {r}. "
                "Need ID alignment before combining."
            )

        arr = np.asarray(emb_r, dtype=np.float32)
        acc += arr
        cnt += 1.0  # per-row same for dense cached reps

        emb_r = None
        gc.collect()

    valid = cnt > 0
    combo = np.zeros((N, D), dtype=np.float32)
    combo[valid] = acc[valid] / cnt[valid, None]

    if normalize:
        denom = np.linalg.norm(combo, axis=1, keepdims=True) + 1e-12
        combo = combo / denom

    emb0 = None
    gc.collect()
    return doc_ids0, combo



def batched_dot_scores(doc_embeddings, q, batch_size=8192):
    """
    Compute scores = doc_embeddings @ q in row batches.
    doc_embeddings: shape (N, D), can be np.memmap
    q: shape (D,), float32
    returns: np.ndarray shape (N,), float32
    """
    n = doc_embeddings.shape[0]
    scores = np.empty(n, dtype=np.float32)

    # ensure query is contiguous float32
    q = np.ascontiguousarray(q, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # block is (B, D), still view-backed for memmap slices
        block = doc_embeddings[start:end]
        # dot per batch
        scores[start:end] = block @ q

    return scores
    
def batched_topk(doc_embeddings, q, k=100, batch_size=8192):
    """
    Exact top-k over batched scoring without storing all scores.
    Returns:
      top_idx: np.ndarray (k,) global doc indices sorted by score desc
      top_scores: np.ndarray (k,) corresponding scores
    """
    n = doc_embeddings.shape[0]
    q = np.ascontiguousarray(q, dtype=np.float32)

    heap = []  # min-heap of (score, idx)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        block_scores = (doc_embeddings[start:end] @ q).astype(np.float32, copy=False)

        for local_i, s in enumerate(block_scores):
            gi = start + local_i
            if len(heap) < k:
                heapq.heappush(heap, (float(s), gi))
            else:
                if s > heap[0][0]:
                    heapq.heapreplace(heap, (float(s), gi))

    # sort descending
    heap.sort(key=lambda x: x[0], reverse=True)
    top_scores = np.array([x[0] for x in heap], dtype=np.float32)
    top_idx = np.array([x[1] for x in heap], dtype=np.int64)
    return top_idx, top_scores




def compute_and_cache_centroid(
    cache_root,
    model_name,
    dataset,
    centroid_name,      # e.g., "centroid_popular"
    rep_list,           # list of base reps included in centroid
):
    if cache_exists(cache_root, model_name, dataset, centroid_name):
        return

    # Load first rep to get doc_ids + shape
    doc_ids0, emb0 = load_cache_mmap(cache_root, model_name, dataset, rep_list[0])
    N, D = emb0.shape

    # Verify doc_ids match across reps (assumes you cached with same table_ids order)
    # If they might differ, you’ll need an ID alignment step; see note below.
    acc = np.zeros((N, D), dtype=np.float32)
    count = 0

    for r in rep_list:
        doc_ids_r, emb_r = load_cache_mmap(cache_root, model_name, dataset, r)
        if doc_ids_r != doc_ids0:
            raise ValueError(f"doc_ids mismatch between {rep_list[0]} and {r}. "
                             f"Need alignment by ID before centroiding.")
        acc += np.asarray(emb_r, dtype=np.float32)
        count += 1

        # release memmap handle reference ASAP
        emb_r = None
        gc.collect()

    acc /= max(count, 1)
    save_cache(cache_root, model_name, dataset, centroid_name, doc_ids0, acc)

    emb0 = None
    gc.collect()

