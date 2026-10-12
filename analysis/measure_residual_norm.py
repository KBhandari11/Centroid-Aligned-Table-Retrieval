"""
measure_residual_norm.py

Validates Assumption 1 (Approximate Perturbation Centering) from the paper by
computing the residual norm ratio rho(T) for every table T, retriever f, and
serialization (sub)family.

For a table T with serialization family S(T):
    z_s(T)     = f(ser_s(T))                       (embedding of serialization s)
    c(T)       = mean over s in S(T) of z_s(T)      (centroid)
    mu(T)      = c(T) - delta_bar(T)                 (stable semantic signal, under Assumption 1)
    delta_s(T) = z_s(T) - c(T)                       (format-specific shift, relative to centroid)
    delta_bar  = mean over s in S(T) of delta_s(T)   (average residual; should be ~0 if Assumption 1 holds)

    rho(T) = ||delta_bar(T)||_2 / ||c(T)||_2

When rho is small, the centering assumption holds. When rho is large, systematic
biases dominate over zero-mean noise.

Additionally, we decompose the residual into:
    mu_delta_s = mean over T of delta_s(T)           (systematic per-format bias, table-independent)
    epsilon_s(T) = delta_s(T) - mu_delta_s            (table-specific random component)

We report both ||mu_delta_s|| and mean ||epsilon_s(T)|| to separate systematic
from random residual components.

Usage:
    python measure_residual_norm.py <model_name> <dataset> [--cache_root PATH] [--out_dir PATH]

Outputs:
    - <out_dir>/<dataset>_<model>_residual_rho_all.npz
    - <out_dir>/<dataset>_<model>_residual_rho_by_subfamily.npz
    - <out_dir>/<dataset>_<model>_systematic_bias.npz
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ============================================================
# REQUIRED: import your loader
# ============================================================
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from utils.cache_embedding import load_cache_for_model
from utils.core import CACHE_ROOT, FORMATS_17

DEFAULT_MAX_TABLES = 10_000
DEFAULT_MAX_TABLES_LARGE = 1_000   # for high-dim models (splade, qwen*)
DEFAULT_SEED = 42

# Models whose embedding dimension is large enough to warrant a smaller default
_LARGE_MODEL_PREFIXES = ("splade", "qwen")


def _is_large_model(model_name: str) -> bool:
    return any(model_name.lower().startswith(p) for p in _LARGE_MODEL_PREFIXES)

# ----------------------------------------------------------------
# Serialization family definitions (chunked pipeline — cache keys
# are prefixed with "chunked_")
# ----------------------------------------------------------------
SERIALIZATION_FAMILIES = {
    "popular": [f"chunked_{r}" for r in ["pipe_serialized", "token_serialized", "space_serialized"]],
    "data": [f"chunked_{r}" for r in ["csv", "tsv", "html", "markdown", "latex", "dict", "json", "xml"]],
    "structural": [f"chunked_{r}" for r in ["shuffled_rows", "shuffled_cols", "transpose"]],
    "schema": [f"chunked_{r}" for r in ["mschema", "macschema", "ddl"]],
}

ALL_REPR_NAMES = [f"chunked_{r}" for r in FORMATS_17]


def _build_table_index(doc_ids: np.ndarray) -> Dict[str, int]:
    """Map each document/table ID to its row index in the embedding matrix."""
    index = {}
    for i, tid in enumerate(doc_ids):
        key = str(tid)
        index[key] = i
    return index


def load_all_embeddings(
    cache_root: str,
    model_name: str,
    dataset: str,
    repr_names: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load chunked embeddings for every requested serialization format and
    aggregate chunk-level vectors into a single mean vector per table.

    Chunked doc_ids are stored as "table_id::chunk_idx".  We group by
    table_id and average all chunk embeddings to produce one (D,) vector
    per table — analogous to the chunk-mean representation used at scoring
    time.

    Returns:
        emb_dict:  {repr_name -> (N_tables, D) float32 array}
        id_dict:   {repr_name -> (N_tables,) array of table IDs}
    """
    emb_dict = {}
    id_dict = {}
    for r in repr_names:
        try:
            chunk_ids, chunk_emb = load_cache_for_model(cache_root, model_name, dataset, r)

            # Group chunk indices by table_id (part before "::")
            table_to_chunks: Dict[str, List[int]] = defaultdict(list)
            for i, cid in enumerate(chunk_ids):
                table_to_chunks[cid.rsplit("::", 1)[0]].append(i)
            table_ids_sorted = sorted(table_to_chunks.keys())

            if hasattr(chunk_emb, 'toarray'):
                # ── Sparse path (SPLADE) ──────────────────────────────────────
                # Average chunk rows per table while keeping the result sparse.
                # This avoids ever materialising the full (N_chunks × vocab) dense
                # matrix; each mean row is immediately re-sparsified.
                import scipy.sparse
                rows = []
                for tid in table_ids_sorted:
                    idxs = table_to_chunks[tid]
                    mean_row = chunk_emb[idxs].mean(axis=0)   # (1, vocab) np.matrix
                    rows.append(scipy.sparse.csr_matrix(mean_row))
                table_emb = scipy.sparse.vstack(rows)          # (N_tables, vocab) CSR
                chunk_emb = None                               # free original sparse
                n_chunks = len(chunk_ids)
                emb_dict[r] = table_emb
                id_dict[r] = np.asarray(table_ids_sorted)
                print(f"  Loaded {r} (sparse): {n_chunks} chunks -> "
                      f"{table_emb.shape}, nnz/row ≈ {table_emb.nnz / max(len(table_ids_sorted), 1):.0f}")
            else:
                # ── Dense path ────────────────────────────────────────────────
                arr = np.asarray(chunk_emb, dtype=np.float32)
                if arr.ndim != 2:
                    print(f"  [SKIP] {r} has shape {arr.shape}, expected 2D")
                    continue
                table_emb = np.stack(
                    [arr[table_to_chunks[tid]].mean(axis=0) for tid in table_ids_sorted],
                    axis=0,
                )  # (N_tables, D)
                emb_dict[r] = table_emb
                id_dict[r] = np.asarray(table_ids_sorted)
                print(f"  Loaded {r}: {len(chunk_ids)} chunks -> {table_emb.shape} table embeddings")
        except Exception as e:
            print(f"  [SKIP] {r}: {e}")
    return emb_dict, id_dict


def find_common_table_ids(
    id_dict: Dict[str, np.ndarray],
    repr_names: List[str],
) -> List[str]:
    """
    Find table IDs present across ALL specified serialization formats.
    """
    available = [r for r in repr_names if r in id_dict]
    if not available:
        return []

    common = set(str(x) for x in id_dict[available[0]])
    for r in available[1:]:
        common &= set(str(x) for x in id_dict[r])

    return sorted(common)

def subsample_table_ids(
    table_ids: List[str],
    max_tables: int,
    seed: int,
) -> List[str]:
    """
    Randomly subsample table IDs down to at most max_tables entries.
    Uses a fixed seed for reproducibility across model-dataset pairs.
    Returns a sorted list.
    """

    if len(table_ids) <= max_tables:
        return table_ids

    rng = np.random.default_rng(seed=seed)
    sampled = rng.choice(table_ids, size=max_tables, replace=False)
    result = sorted(sampled.tolist())
    print(f"Subsampled {len(table_ids)} -> {len(result)} tables (max_tables={max_tables}, seed={seed})")
    return result

def gather_embeddings_by_table(
    emb_dict: Dict[str, np.ndarray],
    id_dict: Dict[str, np.ndarray],
    repr_names: List[str],
    table_ids: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build a nested dict:  table_id -> repr_name -> embedding vector (D,)

    Only includes repr_names that exist in emb_dict.
    """
    available = [r for r in repr_names if r in emb_dict]

    # Build per-repr index lookups once
    repr_indices = {}
    for r in available:
        repr_indices[r] = _build_table_index(id_dict[r])

    result = {}
    for tid in table_ids:
        row = {}
        for r in available:
            idx = repr_indices[r].get(tid, None)
            if idx is not None:
                vec = emb_dict[r][idx]
                # Sparse row (SPLADE) → dense 1-D vector; only one row at a time
                if hasattr(vec, 'toarray'):
                    vec = np.asarray(vec.toarray(), dtype=np.float32).flatten()
                row[r] = vec
        if row:
            result[tid] = row

    return result


def compute_centroid_and_residuals(
    table_embeddings: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Given {repr_name -> embedding} for a single table, compute:
        centroid    (D,)
        delta_bar   (D,)   average format-specific shift (should be ~0)
        deltas      {repr_name -> delta_s(T)}

    Note: delta_bar will always be exactly 0 when computed as mean(z_s - centroid)
    over the FULL family used to compute that centroid. The interesting case arises
    when we compute the centroid over one subfamily and measure residuals relative
    to a different grouping, OR when we examine subfamilies against the global centroid.

    For the "all" centroid case, rho is identically 0 by construction, so we instead
    report the per-subfamily centroid residuals against the global centroid.
    """
    vectors = list(table_embeddings.values())
    names = list(table_embeddings.keys())
    stacked = np.stack(vectors, axis=0)  # (|S|, D)

    centroid = stacked.mean(axis=0)  # (D,)
    deltas = {}
    for name, vec in zip(names, vectors):
        deltas[name] = vec - centroid

    delta_bar = np.stack(list(deltas.values()), axis=0).mean(axis=0)

    return centroid, delta_bar, deltas


def compute_subfamily_centroid_residual(
    table_embeddings: Dict[str, np.ndarray],
    subfamily_members: List[str],
    global_centroid: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Compute the centroid of a subfamily and return its displacement from
    the global centroid.

    This captures systematic subfamily-level bias.
    """
    available = [r for r in subfamily_members if r in table_embeddings]
    if not available:
        return None
    sub_vectors = np.stack([table_embeddings[r] for r in available], axis=0)
    sub_centroid = sub_vectors.mean(axis=0)
    return sub_centroid - global_centroid


def compute_rho(delta_bar: np.ndarray, centroid: np.ndarray, eps: float = 1e-10) -> float:
    """rho(T) = ||delta_bar(T)||_2 / ||c(T)||_2"""
    c_norm = np.linalg.norm(centroid)
    if c_norm < eps:
        return float("nan")
    return float(np.linalg.norm(delta_bar) / c_norm)


def run_residual_analysis(
    cache_root: str,
    model_name: str,
    dataset: str,
    out_dir: str = "./residual_outputs",
    max_tables: Optional[int] = None,
    seed: int = DEFAULT_SEED,
):
    """
    Main analysis pipeline. Computes and saves:

    1. Per-table rho values for each serialization subfamily and the global family.
    2. Systematic per-format bias vectors mu_delta_s.
    3. Per-table random residual magnitudes ||epsilon_s(T)||.

    max_tables defaults to DEFAULT_MAX_TABLES_LARGE (1 000) for large-vocabulary
    models (splade, qwen*) and DEFAULT_MAX_TABLES (10 000) for all others.
    """
    if max_tables is None:
        max_tables = DEFAULT_MAX_TABLES_LARGE if _is_large_model(model_name) else DEFAULT_MAX_TABLES
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Residual Norm Analysis: model={model_name}, dataset={dataset}")
    print(f"{'='*60}")

    # --- Load all embeddings ---
    print("\nLoading embeddings...")
    emb_dict, id_dict = load_all_embeddings(
        cache_root, model_name, dataset, ALL_REPR_NAMES,
    )
    available_reprs = list(emb_dict.keys())
    print(f"\nAvailable representations: {len(available_reprs)}/{len(ALL_REPR_NAMES)}")

    # --- Find tables common to ALL available representations ---
    common_ids = find_common_table_ids(id_dict, available_reprs)
    print(f"Tables with all {len(available_reprs)} representations: {len(common_ids)}")

    if len(common_ids) == 0:
        print("[ERROR] No tables found with embeddings across all representations.")
        return

    # --- Subsample to keep memory and compute tractable ---
    common_ids = subsample_table_ids(common_ids, max_tables=max_tables, seed=seed)

    # --- Gather per-table embeddings ---
    print("Gathering per-table embeddings...")
    per_table = gather_embeddings_by_table(emb_dict, id_dict, available_reprs, common_ids)

    # ================================================================
    # ANALYSIS 1: Subfamily centroid displacement from global centroid
    # ================================================================
    # For each table, compute the global centroid (over all reprs), then
    # measure how far each subfamily's centroid sits from it.
    #
    # rho_subfamily(T) = ||c_sub(T) - c_all(T)||_2 / ||c_all(T)||_2
    # ================================================================
    print("\n--- Analysis 1: Subfamily centroid displacement from global centroid ---")

    subfamily_rho = defaultdict(list)  # family_name -> list of rho values
    global_centroid_norms = []

    for tid in common_ids:
        table_embs = per_table[tid]

        # Global centroid over all available representations
        all_vecs = np.stack(list(table_embs.values()), axis=0)
        global_centroid = all_vecs.mean(axis=0)
        gc_norm = np.linalg.norm(global_centroid)
        global_centroid_norms.append(gc_norm)

        for family_name, family_members in SERIALIZATION_FAMILIES.items():
            displacement = compute_subfamily_centroid_residual(
                table_embs, family_members, global_centroid,
            )
            if displacement is not None and gc_norm > 1e-10:
                rho = float(np.linalg.norm(displacement) / gc_norm)
                subfamily_rho[family_name].append(rho)

    # Save subfamily results
    subfamily_results = {}
    for family_name, rho_list in subfamily_rho.items():
        arr = np.array(rho_list, dtype=np.float64)
        subfamily_results[f"rho_{family_name}"] = arr
        print(f"  {family_name:12s}  median rho = {np.median(arr):.4f}, "
              f"mean = {np.mean(arr):.4f}, std = {np.std(arr):.4f}, "
              f"max = {np.max(arr):.4f}, n = {len(arr)}")

    subfamily_results["table_ids"] = np.array(common_ids, dtype=object)
    subfamily_results["global_centroid_norms"] = np.array(global_centroid_norms)
    subfamily_results["n_tables_sampled"] = np.array(len(common_ids))
    subfamily_results["subsample_seed"] = np.array(seed)

    path_sub = os.path.join(out_dir, f"{dataset}_{model_name}_residual_rho_by_subfamily.npz")
    np.savez(path_sub, **subfamily_results)
    print(f"  Saved -> {path_sub}")

    # ================================================================
    # ANALYSIS 2: Within-subfamily centering check
    # ================================================================
    # For each subfamily independently, compute the subfamily centroid,
    # then measure delta_bar within that subfamily.
    # By construction, delta_bar = 0 within each subfamily relative to
    # its own centroid. So we instead measure the SPREAD of individual
    # deltas: mean ||delta_s(T)|| / ||c_sub(T)||.
    # This captures how scattered the embeddings are around their centroid.
    # ================================================================
    print("\n--- Analysis 2: Within-subfamily embedding spread ---")

    within_spread = defaultdict(list)  # family_name -> list of mean_spread values

    for tid in common_ids:
        table_embs = per_table[tid]

        for family_name, family_members in SERIALIZATION_FAMILIES.items():
            avail = [r for r in family_members if r in table_embs]
            if len(avail) < 2:
                continue
            sub_vecs = np.stack([table_embs[r] for r in avail], axis=0)
            sub_centroid = sub_vecs.mean(axis=0)
            sub_c_norm = np.linalg.norm(sub_centroid)
            if sub_c_norm < 1e-10:
                continue
            delta_norms = np.linalg.norm(sub_vecs - sub_centroid, axis=1)
            mean_spread = float(delta_norms.mean() / sub_c_norm)
            within_spread[family_name].append(mean_spread)

    within_results = {}
    for family_name, spread_list in within_spread.items():
        arr = np.array(spread_list, dtype=np.float64)
        within_results[f"spread_{family_name}"] = arr
        print(f"  {family_name:12s}  median spread = {np.median(arr):.4f}, "
              f"mean = {np.mean(arr):.4f}, std = {np.std(arr):.4f}")

    within_results["table_ids"] = np.array(common_ids, dtype=object)
    path_within = os.path.join(out_dir, f"{dataset}_{model_name}_within_subfamily_spread.npz")
    np.savez(path_within, **within_results)
    print(f"  Saved -> {path_within}")

    # ================================================================
    # ANALYSIS 3: Systematic per-format bias decomposition
    # ================================================================
    # mu_delta_s = mean over T of delta_s(T)      (systematic component)
    # epsilon_s(T) = delta_s(T) - mu_delta_s       (random component)
    #
    # delta_s(T) is computed relative to the GLOBAL centroid c(T).
    #
    # We report ||mu_delta_s|| and mean ||epsilon_s(T)|| for each format s.
    # ================================================================
    print("\n--- Analysis 3: Systematic vs. random residual decomposition ---")

    # Collect all deltas: repr_name -> list of (D,) vectors, one per table
    all_deltas = defaultdict(list)

    for tid in common_ids:
        table_embs = per_table[tid]
        all_vecs = np.stack(list(table_embs.values()), axis=0)
        global_centroid = all_vecs.mean(axis=0)

        for r_name, vec in table_embs.items():
            delta = vec - global_centroid
            all_deltas[r_name].append(delta)

    systematic_bias_norms = {}
    random_residual_mean_norms = {}
    systematic_bias_vectors = {}

    for r_name in available_reprs:
        if r_name not in all_deltas or len(all_deltas[r_name]) == 0:
            continue
        delta_stack = np.stack(all_deltas[r_name], axis=0)  # (N_tables, D)

        # Systematic bias: mean delta across tables
        mu_delta_s = delta_stack.mean(axis=0)  # (D,)
        mu_norm = float(np.linalg.norm(mu_delta_s))

        # Random residual: table-specific deviation from systematic bias
        epsilon = delta_stack - mu_delta_s[np.newaxis, :]  # (N_tables, D)
        eps_norms = np.linalg.norm(epsilon, axis=1)  # (N_tables,)
        mean_eps_norm = float(eps_norms.mean())

        systematic_bias_norms[r_name] = mu_norm
        random_residual_mean_norms[r_name] = mean_eps_norm
        systematic_bias_vectors[r_name] = mu_delta_s

        print(f"  {r_name:20s}  ||mu_delta_s|| = {mu_norm:.4f}, "
              f"mean||epsilon_s|| = {mean_eps_norm:.4f}, "
              f"ratio(sys/rand) = {mu_norm / (mean_eps_norm + 1e-10):.4f}")

    # Save systematic bias results
    bias_results = {
        "repr_names": np.array(list(systematic_bias_norms.keys()), dtype=object),
        "systematic_bias_norms": np.array(
            [systematic_bias_norms[r] for r in systematic_bias_norms], dtype=np.float64,
        ),
        "random_residual_mean_norms": np.array(
            [random_residual_mean_norms[r] for r in random_residual_mean_norms], dtype=np.float64,
        ),
        "n_tables_sampled": np.array(len(common_ids)),
        "subsample_seed": np.array(seed),
    }
    # Also save the full systematic bias vectors for potential downstream use
    for r_name, vec in systematic_bias_vectors.items():
        bias_results[f"mu_delta_{r_name}"] = vec

    path_bias = os.path.join(out_dir, f"{dataset}_{model_name}_systematic_bias.npz")
    np.savez(path_bias, **bias_results)
    print(f"  Saved -> {path_bias}")

    # ================================================================
    # ANALYSIS 4: Cross-subfamily pairwise centroid distances
    # ================================================================
    # For each table, compute each subfamily's centroid and measure
    # pairwise L2 distances between subfamily centroids, normalized by
    # the global centroid norm.
    # ================================================================
    print("\n--- Analysis 4: Cross-subfamily pairwise centroid distances ---")

    family_names = list(SERIALIZATION_FAMILIES.keys())
    n_families = len(family_names)
    pairwise_distances = np.zeros((len(common_ids), n_families, n_families), dtype=np.float64)

    for t_idx, tid in enumerate(common_ids):
        table_embs = per_table[tid]
        all_vecs = np.stack(list(table_embs.values()), axis=0)
        gc_norm = np.linalg.norm(all_vecs.mean(axis=0))
        if gc_norm < 1e-10:
            pairwise_distances[t_idx] = np.nan
            continue

        sub_centroids = {}
        for fi, family_name in enumerate(family_names):
            avail = [r for r in SERIALIZATION_FAMILIES[family_name] if r in table_embs]
            if avail:
                sub_centroids[fi] = np.stack([table_embs[r] for r in avail], axis=0).mean(axis=0)

        for fi in range(n_families):
            for fj in range(fi + 1, n_families):
                if fi in sub_centroids and fj in sub_centroids:
                    dist = np.linalg.norm(sub_centroids[fi] - sub_centroids[fj]) / gc_norm
                    pairwise_distances[t_idx, fi, fj] = dist
                    pairwise_distances[t_idx, fj, fi] = dist

    # Report mean pairwise distances
    mean_pw = pairwise_distances.mean(axis=0)
    for fi in range(n_families):
        for fj in range(fi + 1, n_families):
            print(f"  {family_names[fi]:12s} <-> {family_names[fj]:12s}  "
                  f"mean normalized dist = {mean_pw[fi, fj]:.4f}")

    path_pw = os.path.join(out_dir, f"{dataset}_{model_name}_pairwise_subfamily_distances.npz")
    np.savez(
        path_pw,
        pairwise_distances=pairwise_distances,
        family_names=np.array(family_names, dtype=object),
        table_ids=np.array(common_ids, dtype=object),
        n_tables_sampled=np.array(len(common_ids)),
        subsample_seed=np.array(seed),
    )
    print(f"  Saved -> {path_pw}")

    print(f"\n{'='*60}")
    print("Analysis complete.")
    print(f"{'='*60}\n")


def run_all_models_datasets(
    cache_root: str,
    out_dir: str = "./residual_outputs",
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    max_tables: Optional[int] = None,
    seed: int = DEFAULT_SEED,
):
    """
    Convenience function to run the analysis across all model-dataset pairs.
    """
    if models is None:
        models = ["qwen4B"]#["qwen0.6B"]#["mdbr"]#["splade","miniLM",]#["mpnet", "bge", "reasonir", "splade"]
    if datasets is None:
        datasets = ["WTQ", "WIKISQL", "NQ"]

    for model_name in models:
        for dataset in datasets:
            try:
                run_residual_analysis(
                    cache_root=cache_root,
                    model_name=model_name,
                    dataset=dataset,
                    out_dir=out_dir,
                    max_tables=max_tables,
                    seed=seed,
                )
            except Exception as e:
                print(f"[FAILED] model={model_name}, dataset={dataset}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure residual norm rho(T) to validate Assumption 1."
    )
    parser.add_argument("model_name", type=str,
                        help="Retriever model name (e.g. mpnet, bge, reasonir, splade)")
    parser.add_argument("dataset", type=str,
                        help="Dataset name (e.g. WTQ, WIKISQL, NQ)")
    parser.add_argument("--cache_root", type=str,
                        default=CACHE_ROOT,
                        help="Root directory for cached embeddings")
    parser.add_argument("--out_dir", type=str,
                        default="/data/Kushal/UniversalRetrieval_data/residual_outputs",
                        help="Directory to save analysis results")
    parser.add_argument("--max_tables", type=int,
                        default=DEFAULT_MAX_TABLES,
                        help=f"Maximum number of tables to sample (default: {DEFAULT_MAX_TABLES})")
    parser.add_argument("--seed", type=int,
                        default=DEFAULT_SEED,
                        help=f"Random seed for table subsampling (default: {DEFAULT_SEED})")
    parser.add_argument("--all", action="store_true",
                        help="Run analysis for all model-dataset combinations")

    args = parser.parse_args()
  
    if args.all:
        run_all_models_datasets(
            cache_root=args.cache_root,
            out_dir=args.out_dir,
            max_tables=args.max_tables,
            seed=args.seed,
        )
    else:
        run_residual_analysis(
            cache_root=args.cache_root,
            model_name=args.model_name,
            dataset=args.dataset,
            out_dir=args.out_dir,
            max_tables=args.max_tables,
            seed=args.seed,
        )

    #PYTHONPATH=. python ./analysis/measure_residual_norm.py mpnet WTQ --all > test.txt