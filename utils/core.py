
import gc
import hashlib
import json
import logging
import os
import pickle
import torch
import heapq
import numpy as np
import pandas as pd
from .cache_embedding import (
    cache_exists, save_cache, load_cache_mmap,
    sparse_cache_exists, save_sparse_cache, load_sparse_cache,
)

logger = logging.getLogger(__name__)

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
CACHE_ROOT = "/data/Kushal/UniversalRetrieval_data/emb_cache"

SPLADE_TOPK: int = 256  # Top-K activations kept per SPLADE vector


# ── SPLADE sparse helpers ─────────────────────────────────────────────────────

def _topk_dense_to_sparse(dense: np.ndarray, topk: int):
    """Convert (N, V) dense SPLADE output to scipy CSR, keeping top-K per row."""
    import scipy.sparse
    N, V = dense.shape
    rows, cols, vals = [], [], []
    for i in range(N):
        row = dense[i]
        nz_idx = np.flatnonzero(row > 0)
        if len(nz_idx) > topk:
            top_pos = np.argpartition(row[nz_idx], -topk)[-topk:]
            nz_idx = nz_idx[top_pos]
        if len(nz_idx):
            rows.extend([i] * len(nz_idx))
            cols.extend(nz_idx.tolist())
            vals.extend(row[nz_idx].tolist())
    return scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(N, V), dtype=np.float32
    )


def _topk_sparse_rows(csr, topk: int):
    """Keep top-K nonzero values per row in a scipy CSR matrix (in sparse space)."""
    import scipy.sparse
    csr = csr.tocsr()
    data = csr.data.copy().astype(np.float32)
    for i in range(csr.shape[0]):
        s, e = csr.indptr[i], csr.indptr[i + 1]
        seg = data[s:e]
        if len(seg) > topk:
            threshold_pos = np.argpartition(seg, -topk)[-topk:]
            mask = np.zeros(len(seg), dtype=bool)
            mask[threshold_pos] = True
            seg[~mask] = 0.0
            data[s:e] = seg
    result = scipy.sparse.csr_matrix(
        (data, csr.indices.copy(), csr.indptr.copy()),
        shape=csr.shape, dtype=np.float32,
    )
    result.eliminate_zeros()
    return result


def _splade_encode_to_sparse(model, batch, topk: int = SPLADE_TOPK):
    """Encode a batch with SPLADE, apply top-K, return scipy CSR (batch_size, vocab)."""
    dense = model.encode_document(batch).to_dense()
    if isinstance(dense, torch.Tensor):
        dense = dense.detach().cpu().float().numpy()
    else:
        dense = np.asarray(dense, dtype=np.float32)
    return _topk_dense_to_sparse(dense, topk)


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

        if model_name == "splade":
            # Keep sparse: encode and apply top-K, returns scipy CSR
            batch_emb = _splade_encode_to_sparse(model, batch)
            all_emb.append(batch_emb)
            continue

        if model_name == "reasonir":
            emb = model.encode(batch, instruction=instruction)
        elif model_name in ("bge", "mpnet"):
            emb = model.encode(batch)
        elif model_name == "jina":
            emb = model.encode(batch, task="retrieval.passage", prompt_name="retrieval.passage")
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

    if model_name == "splade":
        import scipy.sparse
        sparse_mat = scipy.sparse.vstack(all_emb, format="csr")
        save_sparse_cache(cache_root, model_name, dataset, rep_name, doc_ids, sparse_mat)
        return doc_ids, sparse_mat

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
    doc_embeddings: (N, D) np.memmap  OR  scipy CSR sparse matrix (for SPLADE)
    q: shape (D,), float32
    returns: np.ndarray shape (N,), float32
    """
    import scipy.sparse
    q = np.ascontiguousarray(q, dtype=np.float32)

    if scipy.sparse.issparse(doc_embeddings):
        # Sparse matrix @ dense vector — scipy handles this efficiently
        scores = np.asarray(doc_embeddings.dot(q), dtype=np.float32).ravel()
        return scores

    n = doc_embeddings.shape[0]
    scores = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        block = doc_embeddings[start:end]
        scores[start:end] = block @ q
    return scores


def batched_topk(doc_embeddings, q, k=100, batch_size=8192):
    """
    Exact top-k over batched scoring without storing all scores.
    doc_embeddings: (N, D) np.memmap  OR  scipy CSR sparse matrix (for SPLADE)
    Returns:
      top_idx: np.ndarray (k,) global doc indices sorted by score desc
      top_scores: np.ndarray (k,) corresponding scores
    """
    import scipy.sparse
    q = np.ascontiguousarray(q, dtype=np.float32)

    if scipy.sparse.issparse(doc_embeddings):
        scores = batched_dot_scores(doc_embeddings, q)
        n = len(scores)
        actual_k = min(k, n)
        if actual_k == n:
            top_idx = np.argsort(scores)[::-1].astype(np.int64)
        else:
            top_idx = np.argpartition(scores, -actual_k)[-actual_k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return top_idx.astype(np.int64), scores[top_idx].astype(np.float32)

    n = doc_embeddings.shape[0]
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
    # ── SPLADE: work entirely in sparse space, re-sparsify with top-K ─────────
    if model_name == "splade":
        if sparse_cache_exists(cache_root, model_name, dataset, centroid_name):
            return

        acc = None
        doc_ids0 = None
        count = 0

        for r in rep_list:
            if not sparse_cache_exists(cache_root, model_name, dataset, r):
                continue
            doc_ids_r, mat_r = load_sparse_cache(cache_root, model_name, dataset, r)
            mat_r = mat_r.astype(np.float32)
            if count == 0:
                doc_ids0 = doc_ids_r
                acc = mat_r
            else:
                if doc_ids_r != doc_ids0:
                    raise ValueError(
                        f"doc_ids mismatch between {rep_list[0]} and {r}. "
                        "Need alignment by ID before centroiding."
                    )
                acc = acc + mat_r
            count += 1
            mat_r = None
            gc.collect()

        if acc is None or count == 0:
            return

        # Average then project back to sparse space via top-K
        acc = acc / count
        acc = _topk_sparse_rows(acc, SPLADE_TOPK)
        save_sparse_cache(cache_root, model_name, dataset, centroid_name, doc_ids0, acc)
        acc = None
        gc.collect()
        return

    # ── Dense path (all other models) ─────────────────────────────────────────
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


# =============================================================================
# Chunked embedding pipeline  (Steps 1-8)
# =============================================================================

# ── Constants ─────────────────────────────────────────────────────────────────

CELL_CHAR_BUDGET: int = 100        # Step 1 – chars per cell value
TOKEN_BUDGET: int = 472           # Step 3 – 512 minus 40-token reserve
MAX_ROWS_SEARCH: int = 50         # Step 3 – binary-search upper bound

FORMATS_17 = [
    "pipe_serialized", "token_serialized", "space_serialized",
    "csv", "tsv", "html", "markdown", "latex", "dict", "json", "xml",
    "shuffled_rows", "shuffled_cols", "transpose",
    "mschema", "macschema", "ddl",
]

# ── Step 1: Normalize ─────────────────────────────────────────────────────────

def truncate_df(df: pd.DataFrame, char_budget: int = CELL_CHAR_BUDGET) -> pd.DataFrame:
    """
    Step 1: Truncate every cell to char_budget characters.
    Every serializer receives identical, pre-normalized content.
    Returns a copy; the original is untouched.
    """
    def _trunc(x):
        return "" if pd.isna(x) else str(x)[:char_budget]

    try:
        return df.map(_trunc)          # pandas >= 2.1
    except AttributeError:
        return df.applymap(_trunc)     # pandas < 2.1


# ── Schema preambles (repeated every chunk for schema formats) ─────────────────

def _preamble_ddl(df: pd.DataFrame, table_name: str = "Table") -> str:
    dtype_map = {
        "int64": "INTEGER", "float64": "FLOAT", "object": "TEXT",
        "bool": "BOOLEAN", "datetime64[ns]": "TIMESTAMP",
    }
    ddl = f"CREATE TABLE {table_name} (\n"
    for col in df.columns:
        ddl += f"  {col} {dtype_map.get(str(df[col].dtype), 'TEXT')},\n"
    return ddl.rstrip(",\n") + "\n);\n"


def _preamble_mschema(df: pd.DataFrame) -> str:
    return "Schema:\n" + "".join(f"  {c}: {df[c].dtype}\n" for c in df.columns)


def _preamble_macschema(df: pd.DataFrame) -> str:
    import pandas.io.json as pij
    schema = pij.build_table_schema(df, index=False)
    fields = "; ".join(
        f"{f['name']}({f.get('type', '?')})" for f in schema.get("fields", [])
    )
    return f"Fields: {fields}\n"


# ── Step 5 & 6: Render one chunk ──────────────────────────────────────────────

def render_chunk(
    df_full: pd.DataFrame,
    df_chunk: pd.DataFrame,
    format_name: str,
    row_start: int,
    row_end: int,
    title: str = "",
    section: str = "",
) -> str:
    """
    Step 5: header block (title/section + row-range marker + schema preamble
    for schema formats) followed by the rows in the native format.

    Step 6: for 'transpose', block-transpose only df_chunk (same original rows
    as csv chunk k, but transposed within that block).

    For 'shuffled_rows'/'shuffled_cols', rows/columns are randomly shuffled
    within the chunk before rendering as csv.
    """
    meta = ""
    if title:
        meta += f"Title: {title}"
    if section:
        meta += f"  Section: {section}"
    if meta:
        meta += "\n"
    meta += f"[Rows {row_start}-{row_end - 1}]\n"

    preamble = ""
    if format_name == "ddl":
        preamble = _preamble_ddl(df_full)
    elif format_name == "mschema":
        preamble = _preamble_mschema(df_full)
    elif format_name == "macschema":
        preamble = _preamble_macschema(df_full)

    return meta + preamble + _render_body(df_chunk, format_name)


def _safe_xml_tag(name: str) -> str:
    """Convert any string to a valid XML element name."""
    import re
    s = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(name))
    # XML names must start with a letter or underscore
    if not s or not (s[0].isalpha() or s[0] == '_'):
        s = 'col_' + s
    return s


def _render_body(df_chunk: pd.DataFrame, format_name: str) -> str:
    if format_name == "transpose":
        # Step 6: block-transpose — only this chunk's rows
        t = df_chunk.T.copy()
        t.columns = [f"r{i}" for i in range(len(df_chunk))]
        return t.to_csv()

    if format_name == "shuffled_rows":
        return df_chunk.sample(frac=1, random_state=42).reset_index(drop=True).to_csv(index=False)

    if format_name == "shuffled_cols":
        cols = list(df_chunk.columns)
        import random as _random
        _random.seed(42)
        _random.shuffle(cols)
        return df_chunk[cols].to_csv(index=False)

    if format_name == "csv":
        return df_chunk.to_csv(index=False)

    if format_name == "tsv":
        return df_chunk.to_csv(sep="\t", index=False)

    if format_name == "html":
        return df_chunk.to_html(index=False)

    if format_name == "markdown":
        return df_chunk.to_markdown(index=False)

    if format_name == "latex":
        try:
            return df_chunk.to_latex(index=False)
        except Exception:
            return df_chunk.to_csv(index=False)

    if format_name == "dict":
        return str(df_chunk.to_dict(orient="records"))

    if format_name == "json":
        try:
            return df_chunk.to_json(orient="records")
        except Exception:
            return "[]"

    if format_name == "xml":
        try:
            # Sanitize column names: XML tag names cannot contain spaces,
            # newlines, #, /, hyphens at the start, etc.
            safe = df_chunk.copy()
            safe.columns = [_safe_xml_tag(c) for c in df_chunk.columns]
            return safe.to_xml(index=False, pretty_print=False)
        except Exception:
            return "<table/>"

    if format_name == "pipe_serialized":
        hdr = " | ".join(str(c) for c in df_chunk.columns)
        rows = "\n".join(
            " | ".join(str(v) for v in row)
            for row in df_chunk.itertuples(index=False)
        )
        return hdr + "\n" + rows

    if format_name == "token_serialized":
        parts = [f"<Header, 0, {i}> {c}" for i, c in enumerate(df_chunk.columns)]
        for ri, row in enumerate(df_chunk.itertuples(index=False)):
            for ci, val in enumerate(row):
                parts.append(f"<CellValue, {ri + 1}, {ci}> {val}")
        return " ".join(parts)

    if format_name == "space_serialized":
        hdr = " ".join(str(c) for c in df_chunk.columns)
        rows = " ".join(
            " ".join(str(v) for v in row)
            for row in df_chunk.itertuples(index=False)
        )
        return hdr + " " + rows

    if format_name == "ddl":
        out = ""
        for row in df_chunk.itertuples(index=False):
            vals = []
            for v in row:
                if pd.isnull(v):
                    vals.append("NULL")
                elif isinstance(v, bool):
                    vals.append(str(int(v)))
                elif isinstance(v, str):
                    vals.append("'{}'".format(v.replace("'", "''")))
                else:
                    vals.append(str(v))
            out += f"INSERT INTO Table VALUES ({', '.join(vals)});\n"
        return out

    if format_name == "mschema":
        return str(df_chunk.to_dict(orient="records"))

    if format_name == "macschema":
        try:
            return df_chunk.to_json(orient="records")
        except Exception:
            return "[]"

    return df_chunk.to_csv(index=False)   # fallback


# ── Tokenizer helpers ─────────────────────────────────────────────────────────

def get_tokenizer_from_model(model, model_name):
    """
    Extract the tokenizer from an already-loaded embedding model.

    Supports:
      - SentenceTransformer  (bge, mpnet, jina)  → model.tokenizer
      - SparseEncoder        (splade)             → model.tokenizer
      - AutoModel            (reasonir, rank1)    → AutoTokenizer from config
      - bm25 / None                               → returns None

    Returns a single-element list [tokenizer] for direct use with
    build_manifest / compute_R, or None for bm25.
    """
    if model is None or model_name == "bm25":
        return None

    # SentenceTransformer and SparseEncoder expose .tokenizer directly
    if hasattr(model, "tokenizer"):
        tok = model.tokenizer
        logger.info(f"Using tokenizer from model.tokenizer ({type(tok).__name__})")
        return [tok]

    # AutoModel: reconstruct from the checkpoint path stored in config
    if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
        from transformers import AutoTokenizer
        name_or_path = model.config._name_or_path
        tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
        logger.info(f"Loaded tokenizer from config._name_or_path: {name_or_path}")
        return [tok]

    raise RuntimeError(
        f"Cannot extract tokenizer from model '{model_name}' "
        f"(type: {type(model).__name__}).  The model exposes neither "
        "'.tokenizer' nor '.config._name_or_path'."
    )


def load_tokenizers(model_names):
    """
    Load an AutoTokenizer for each name in model_names.
    Prefer get_tokenizer_from_model() when a model is already loaded —
    this helper is only needed when you want to load additional tokenizers
    by HuggingFace model name string.
    """
    from transformers import AutoTokenizer
    toks = []
    for name in model_names:
        try:
            toks.append(AutoTokenizer.from_pretrained(name))
            logger.info(f"Loaded tokenizer: {name}")
        except Exception as exc:
            logger.warning(f"Could not load tokenizer '{name}': {exc}")
    if not toks:
        raise RuntimeError("No tokenizers could be loaded.")
    return toks


# ── Step 3: Compute R(T) ──────────────────────────────────────────────────────

def _max_token_count(text: str, tokenizers) -> int:
    mx = 0
    for tok in tokenizers:
        try:
            ids = tok.encode(text, add_special_tokens=True, truncation=False)
            n = len(ids) if isinstance(ids, list) else int(ids.shape[-1])
            if n > mx:
                mx = n
        except Exception:
            pass
    return mx


def _max_tokens_all_formats(df_norm, n_rows, tokenizers, title, section):
    df_chunk = df_norm.iloc[:n_rows]
    mx = 0
    for fmt in FORMATS_17:
        try:
            text = render_chunk(
                df_norm, df_chunk, fmt,
                row_start=0, row_end=n_rows,
                title=title, section=section,
            )
            c = _max_token_count(text, tokenizers)
            if c > mx:
                mx = c
        except Exception:
            pass
    return mx


def compute_R(
    df_norm: pd.DataFrame,
    tokenizers,
    title: str = "",
    section: str = "",
    max_rows: int = MAX_ROWS_SEARCH,
    token_budget: int = TOKEN_BUDGET,
) -> int:
    """
    Step 3: Binary search downward from max_rows.
    Returns the largest R in [1, min(max_rows, nrows)] such that the
    worst-case token count across all 17 formats and all supplied tokenizers
    fits within token_budget.
    Returns 0 if even R=1 exceeds the budget (table is dropped).
    tokenizers: list of AutoTokenizer objects for the models you intend to use.
    """
    nrows = len(df_norm)
    hi = min(max_rows, nrows)
    lo = 1

    if _max_tokens_all_formats(df_norm, hi, tokenizers, title, section) <= token_budget:
        return hi

    result = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        count = _max_tokens_all_formats(df_norm, mid, tokenizers, title, section)
        if count <= token_budget:
            result = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return result


# ── Step 4: Manifest ──────────────────────────────────────────────────────────

def build_manifest(
    table_corpus,
    dataset: str,
    tokenizers,
    char_budget: int = CELL_CHAR_BUDGET,
    max_rows: int = MAX_ROWS_SEARCH,
    token_budget: int = TOKEN_BUDGET,
    manifest_path=None,
) -> dict:
    """
    Step 4: Compute R(T) for every table, record chunk boundaries, and write
    one JSON manifest per dataset.

    tokenizers: list of AutoTokenizer objects for all models you plan to use
                so R is conservative across all of them.

    Every downstream job reads this file; nothing recomputes boundaries.
    Manifest structure:
      {
        "dataset": str,
        "cell_char_budget": int,
        "token_budget": int,
        "tables": {
          "<table_id>": {
            "R": int,
            "chunk_count": int,
            "chunks": [{"chunk_id": k, "row_start": s, "row_end": e}, ...]
          }
        },
        "dropped_tables": [...]
      }
    """
    manifest = {
        "dataset": dataset,
        "cell_char_budget": char_budget,
        "token_budget": token_budget,
        "tables": {},
        "dropped_tables": [],
    }
    dropped = []

    for table_id, table_entry in table_corpus.items():
        df_raw = table_entry.get("df_string")
        if df_raw is None or not isinstance(df_raw, pd.DataFrame):
            continue

        title = str(table_id)

        # Step 1: normalize cell lengths
        df_norm = truncate_df(df_raw, char_budget)
        nrows, ncols = len(df_norm), len(df_norm.columns)
        if nrows == 0 or ncols == 0:
            dropped.append(str(table_id))
            continue

        # Step 3: compute R via binary search (no permutation)
        R = compute_R(
            df_norm, tokenizers,
            title=title, section="",
            max_rows=max_rows, token_budget=token_budget,
        )

        if R == 0:
            # Single row still exceeds budget (very wide table).
            # Force R=1 and keep the table in the index; the embedding model's
            # built-in max-length handling will hard-truncate overlong formats
            # for smaller-context models (e.g. BERT/MPNet at 512 tokens).
            R = 1

        chunks = [
            {"chunk_id": k, "row_start": s, "row_end": min(s + R, nrows)}
            for k, s in enumerate(range(0, nrows, R))
        ]

        manifest["tables"][str(table_id)] = {
            "R": R,
            "chunk_count": len(chunks),
            "chunks": chunks,
        }

    manifest["dropped_tables"] = dropped
    logger.info(
        f"[{dataset}] manifest: {len(manifest['tables'])} tables, "
        f"{len(dropped)} dropped (empty/invalid)"
    )

    if manifest_path:
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        logger.info(f"Manifest written → {manifest_path}")

    return manifest


def load_manifest(manifest_path: str) -> dict:
    with open(manifest_path) as f:
        return json.load(f)


# ── Steps 5/6: Iterate chunk texts ────────────────────────────────────────────

def iter_chunk_texts(table_corpus, manifest, format_name):
    """
    Yield (doc_id_str, chunk_text) for every chunk of every table.
    doc_id_str = "<table_id>::<chunk_id>".

    Normalization (Step 1) is driven by manifest values so the content is
    identical across all formats.
    """
    char_budget = manifest.get("cell_char_budget", CELL_CHAR_BUDGET)

    for table_id_str, tinfo in manifest["tables"].items():
        entry = table_corpus.get(table_id_str)
        if entry is None:
            continue
        df_raw = entry.get("df_string")
        if df_raw is None or not isinstance(df_raw, pd.DataFrame):
            continue

        df_norm = truncate_df(df_raw, char_budget)

        for chunk_info in tinfo["chunks"]:
            cid = chunk_info["chunk_id"]
            s, e = chunk_info["row_start"], chunk_info["row_end"]
            df_chunk = df_norm.iloc[s:e]

            try:
                text = render_chunk(
                    df_norm, df_chunk, format_name,
                    row_start=s, row_end=e,
                    title=table_id_str, section="",
                )
            except Exception as exc:
                logger.warning(
                    f"render_chunk failed: {table_id_str} chunk {cid} "
                    f"format {format_name}: {exc}"
                )
                text = ""

            yield f"{table_id_str}::{cid}", (text.strip() or "[EMPTY]")


# ── Step 7: Encode and cache chunked embeddings ───────────────────────────────

def compute_and_cache_chunked_embeddings(
    model_name,
    model,
    table_corpus,
    manifest,
    format_name,
    cache_root,
    dataset,
    batch_size=256,
    normalize=False,
    instruction="",
):
    """
    Step 7: Encode all chunks for one format and cache them.

    Cache key = "chunked_{format_name}"
    doc_ids = list of "<table_id>::<chunk_id>" strings
    One vector per (table_id, format, chunk_id).
    Queries stay unchunked and unmodified.

    If CUDA runs out of memory, reduce the batch size by half
    and retry the failed batch.
    """
    rep_key = f"chunked_{format_name}"

    if model_name == "splade":
        if sparse_cache_exists(cache_root, model_name, dataset, rep_key):
            logger.info(f"[{dataset}] chunked {format_name} sparse cache hit, skipping")
            return load_sparse_cache(cache_root, model_name, dataset, rep_key)
    elif cache_exists(cache_root, model_name, dataset, rep_key):
        logger.info(f"[{dataset}] chunked {format_name} cache hit, skipping serialization and encoding")
        return load_cache_mmap(cache_root, model_name, dataset, rep_key)

    doc_ids, docs = [], []
    for doc_id_str, text in iter_chunk_texts(table_corpus, manifest, format_name):
        doc_ids.append(doc_id_str)
        docs.append(text)

    if not doc_ids:
        return None, None

    all_emb = []
    current_batch_size = batch_size
    i = 0

    while i < len(docs):
        batch = docs[i:i + current_batch_size]

        try:
            if model_name == "splade":
                # Keep sparse: apply top-K, returns scipy CSR directly
                batch_emb = _splade_encode_to_sparse(model, batch)
            elif model_name == "reasonir":
                emb = model.encode(batch, instruction=instruction)
            elif model_name in ("bge", "mpnet"):
                emb = model.encode(batch)
            elif model_name == "jina":
                emb = model.encode(batch, task="retrieval.passage", prompt_name="retrieval.passage")
            else:
                emb = model.encode(batch)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

            if current_batch_size == 1:
                logger.error(f"[{dataset}] CUDA OOM with batch_size=1 for model={model_name}")
                raise

            new_batch_size = max(1, current_batch_size // 2)

            logger.warning(
                f"[{dataset}] CUDA OOM for model={model_name} at batch_size={current_batch_size}. "
                f"Retrying with batch_size={new_batch_size}"
            )

            current_batch_size = new_batch_size
            continue

        if model_name != "splade":
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().float().numpy()
            else:
                emb = np.asarray(emb, dtype=np.float32)

            if normalize:
                denom = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
                emb = emb / denom

            batch_emb = emb

        all_emb.append(batch_emb)
        i += len(batch)

    if model_name == "splade":
        import scipy.sparse
        sparse_mat = scipy.sparse.vstack(all_emb, format="csr")
        save_sparse_cache(cache_root, model_name, dataset, rep_key, doc_ids, sparse_mat)
        return doc_ids, sparse_mat

    emb_np = np.concatenate(all_emb, axis=0).astype(np.float32, copy=False)
    save_cache(cache_root, model_name, dataset, rep_key, doc_ids, emb_np)
    return doc_ids, emb_np


# ── Step 8: Score by top-3 chunk mean ─────────────────────────────────────────

def score_top3_chunk_mean(doc_ids, doc_embeddings, q, k=100, batch_size=8192):
    """
    Step 8: Collapse chunk-level scores to a single score per table.

    1. Dot-product score every chunk embedding vs q.
    2. Group by table_id (part before "::").
    3. Primary  = mean of top-3 chunk scores per table.
       Secondary = max chunk score per table (reported alongside).
    4. Sort by primary descending; return top-k
       [(table_id, primary, secondary), ...].
    """
    chunk_scores = batched_dot_scores(doc_embeddings, q, batch_size=batch_size)

    from collections import defaultdict
    tbl_scores = defaultdict(list)
    for did, sc in zip(doc_ids, chunk_scores):
        tbl_scores[did.rsplit("::", 1)[0]].append(float(sc))

    results = []
    for table_id, scores in tbl_scores.items():
        desc = sorted(scores, reverse=True)
        primary = sum(desc[:3]) / len(desc[:3])
        results.append((table_id, primary, desc[0]))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]


# ── Sanity check (run before any encoding) ────────────────────────────────────

def sanity_check_chunk_alignment(table_corpus, manifest, n_tables=10) -> bool:
    """
    Pick n_tables tables, print chunk 0 of csv and xml side by side, and
    confirm they contain the same rows.  Returns True if all pass.
    If any fail the manifest is wrong and every downstream number is
    meaningless.
    """
    table_ids = list(manifest["tables"].keys())[:n_tables]
    all_ok = True
    char_budget = manifest.get("cell_char_budget", CELL_CHAR_BUDGET)

    for tid in table_ids:
        tinfo = manifest["tables"][tid]
        entry = table_corpus.get(tid)
        if entry is None or not isinstance(entry.get("df_string"), pd.DataFrame):
            print(f"[SKIP] {tid}")
            continue

        df_norm = truncate_df(entry["df_string"], char_budget)
        chunk0 = tinfo["chunks"][0]
        s, e = chunk0["row_start"], chunk0["row_end"]
        df_chunk = df_norm.iloc[s:e]

        csv_text = render_chunk(df_norm, df_chunk, "csv", s, e, tid, "")
        xml_text = render_chunk(df_norm, df_chunk, "xml", s, e, tid, "")

        print(f"\n{'=' * 60}")
        print(f"Table: {tid}  R={tinfo['R']}  chunk0 rows [{s}-{e - 1}]")
        print(f"--- CSV chunk 0 ---\n{csv_text[:500]}")
        print(f"--- XML chunk 0 ---\n{xml_text[:500]}")

        mismatch = any(
            str(v)[:20] and (str(v)[:20] not in csv_text or str(v)[:20] not in xml_text)
            for row in df_chunk.itertuples(index=False)
            for v in row
        )

        if mismatch:
            print(f"[MISMATCH] {tid}: rows differ between csv and xml!")
            all_ok = False
        else:
            print(f"[OK] {tid}")

    return all_ok
