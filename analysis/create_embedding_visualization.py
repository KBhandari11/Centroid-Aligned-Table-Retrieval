import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from typing import Dict, List, Tuple, Iterable, Optional


# ============================================================
# REQUIRED: import your loader from retrieval_all_multiple_dataset.py
# ============================================================
# Example:
# from retrieval_all_multiple_dataset import load_cache_mmap
#
# load_cache_mmap(cache_root, model_name, dataset, r)
# should return an mmap-able numpy array (N, D) or equivalent.
from retrieval_all_multiple_dataset import load_cache_mmap


def iter_cache_chunks(
    cache_root: str,
    model_name: str,
    dataset: str,
    repr_names: List[str],
    chunk_size: int = 4096,
) -> Iterable[Tuple[str, np.ndarray]]:
    """
    Yields (repr_name, chunk) where chunk is float32 and shape (n_chunk, d).
    """
    for r in repr_names:
        doc_ids, emb = load_cache_mmap(cache_root, model_name, dataset, r)
        arr = np.asarray(emb, dtype=np.float32)   # only embeddings
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D embeddings for r={r}, got shape={arr.shape}")

        n = arr.shape[0]
        for i in range(0, n, chunk_size):
            chunk = np.asarray(arr[i:i + chunk_size], dtype=np.float32)
            if chunk.shape[0] > 0:
                yield r, chunk


def count_total_rows(
    cache_root: str,
    model_name: str,
    dataset: str,
    repr_names: List[str],
) -> int:
    total = 0
    for r in repr_names:
        doc_ids, emb = load_cache_mmap(cache_root, model_name, dataset, r)
        arr = np.asarray(emb, dtype=np.float32)   # only embeddings
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D embeddings for r={r}, got shape={arr.shape}")
        total += arr.shape[0]
    print(f"Total rows: {total}")
    return total


def fit_global_ipca(
    cache_root: str,
    model_name: str,
    dataset: str,
    repr_names: List[str],
    n_components: int = 2,
    chunk_size: int = 4096,
) -> IncrementalPCA:
    """
    Pass 1: global fit with IncrementalPCA over all chunks.
    """
    ipca = IncrementalPCA(n_components=n_components, batch_size=chunk_size)
    seen = 0
    print("Here:")
    for r, chunk in iter_cache_chunks(cache_root, model_name, dataset, repr_names, chunk_size):
        print("\t\t",r," = ", seen, flush=True)
        # partial_fit requires chunk rows >= n_components
        if chunk.shape[0] >= n_components:
            ipca.partial_fit(chunk)
            seen += chunk.shape[0]

    if seen == 0:
        raise RuntimeError("No chunks with sufficient rows to fit PCA.")

    return ipca


def transform_all_to_memmap(
    cache_root: str,
    model_name: str,
    dataset: str,
    repr_names: List[str],
    ipca: IncrementalPCA,
    out_dir: str,
    n_components: int = 2,
    chunk_size: int = 4096,
) -> Tuple[np.memmap, np.ndarray]:
    """
    Pass 2: transform all chunks and store reduced vectors in memmap.
    Returns:
        Z_memmap: (N_total, n_components), float32
        y_labels: (N_total,), object (repr_name per row)
    """
    os.makedirs(out_dir, exist_ok=True)
    total_rows = count_total_rows(cache_root, model_name, dataset, repr_names)

    z_path = os.path.join(out_dir, f"{dataset}_{model_name}_pca_{n_components}d.memmap")
    Z = np.memmap(z_path, mode="w+", dtype="float32", shape=(total_rows, n_components))
    y = np.empty((total_rows,), dtype=object)

    w = 0
    for r, chunk in iter_cache_chunks(cache_root, model_name, dataset, repr_names, chunk_size):
        z = ipca.transform(chunk).astype(np.float32, copy=False)
        n = z.shape[0]
        Z[w:w+n] = z
        y[w:w+n] = r
        w += n

    Z.flush()
    return Z, y


def balanced_subsample_indices(
    labels: np.ndarray,
    max_per_class: int = 30000,
    seed: int = 42,
) -> np.ndarray:
    """
    Balanced per-representation subsampling for readable plotting.
    """
    rng = np.random.default_rng(seed)
    idx_all = []
    unique = np.unique(labels)
    for c in unique:
        idx = np.where(labels == c)[0]
        if idx.shape[0] > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        idx_all.append(idx)
    return np.concatenate(idx_all) if idx_all else np.array([], dtype=np.int64)


def plot_pca_scatter(
    Z: np.ndarray,
    y: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "PCA projection of cached embeddings",
    s: float = 3.0,
    alpha: float = 0.35,
):
    """
    One color per representation.
    """
    y = np.asarray(y)
    classes = np.unique(y)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(classes))]
    plt.figure(figsize=(12, 9))
    for i, c in enumerate(classes):
        mask = (y == c)
        zc = Z[mask]
        plt.scatter(zc[:, 0], 
                    zc[:, 1], 
                    s=s, 
                    alpha=alpha, 
                    color=colors[i],
                    label=f"{c}")
        # one text label per class at centroid
        cx = zc[:, 0].mean()
        cy = zc[:, 1].mean()
        plt.text(
            cx,
            cy,
            str(c),
            fontsize=11,
            ha="center",
            va="center",
            #bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8, markerscale=3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    else:
        plt.show()


# ------------------------------------------------------------
# Optional: what you asked ("separate PCA then concat")
# WARNING: not geometrically comparable across chunks.
# ------------------------------------------------------------
def separate_pca_per_chunk_then_concat(
    cache_root: str,
    model_name: str,
    dataset: str,
    repr_names: List[str],
    chunk_size: int = 4096,
    n_components: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This fits PCA independently for each chunk, then concatenates.
    NOT recommended for global comparison.
    """
    from sklearn.decomposition import PCA

    z_list = []
    y_list = []
    for r, chunk in iter_cache_chunks(cache_root, model_name, dataset, repr_names, chunk_size):
        
        if chunk.shape[0] < n_components:
            continue
        pca = PCA(n_components=n_components)
        z = pca.fit_transform(chunk).astype(np.float32)
        z_list.append(z)
        y_list.extend([r] * z.shape[0])

    if not z_list:
        raise RuntimeError("No chunk produced PCA output.")
    Z = np.concatenate(z_list, axis=0)
    y = np.array(y_list, dtype=object)
    return Z, y


def run(
    cache_root: str,
    model_name: str,
    dataset: str,
    repr_names: List[str],
    out_dir: str = "./pca_outputs",
    n_components: int = 2,
    chunk_size: int = 4096,
    max_plot_per_repr: int = 30000,
    seed: int = 42,
):
    print("Fitting global IncrementalPCA...")
    ipca = fit_global_ipca(
        cache_root=cache_root,
        model_name=model_name,
        dataset=dataset,
        repr_names=repr_names,
        n_components=n_components,
        chunk_size=chunk_size,
    )
    print("Explained variance ratio:", ipca.explained_variance_ratio_)

    print("Transforming all embeddings...")
    Z_memmap, y = transform_all_to_memmap(
        cache_root=cache_root,
        model_name=model_name,
        dataset=dataset,
        repr_names=repr_names,
        ipca=ipca,
        out_dir=out_dir,
        n_components=n_components,
        chunk_size=chunk_size,
    )

    # Subsample only for plotting readability/speed
    plot_idx = balanced_subsample_indices(y, max_per_class=max_plot_per_repr, seed=seed)
    if plot_idx.size == 0:
        raise RuntimeError("No points selected for plotting.")

    Z_plot = np.asarray(Z_memmap[plot_idx])
    y_plot = y[plot_idx]

    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"{dataset}_{model_name}_pca_scatter.png")
    plot_pca_scatter(
        Z=Z_plot,
        y=y_plot,
        save_path=fig_path,
        title=f"PCA (global IncrementalPCA) - {dataset} - {model_name}",
        s=3.0,
        alpha=0.35,
    )
    print("Done.")


if __name__ == "__main__":
    CACHE_ROOT = "/data/Kushal/UniversalRetrieval_data/emb_cache" 
    MODEL_NAME = sys.argv[1]
    DATASET = sys.argv[2]
    REPR_NAMES =  ["pipe_serialized", "token_serialized", "space_serialized",
                  "csv", "tsv", "html", "markdown", "latex", "dict", "json", "xml",
                  "shuffled_rows", "shuffled_cols", "transpose",
                  "mschema", "macschema", "ddl"
                  ]

    run(
        cache_root=CACHE_ROOT,
        model_name=MODEL_NAME,
        dataset=DATASET,
        repr_names=REPR_NAMES,
        out_dir="./pca_outputs",
        n_components=2,
        chunk_size=4096,
        max_plot_per_repr=10000,
        seed=42,
    )
#CUDA_VISIBLE_DEVICES=1 python create_embedding_visualization.py reasonir WTQ > result_reasonir.txt 2>&1
#CUDA_VISIBLE_DEVICES=1 python create_embedding_visualization.py mpnet WTQ > result_mpnet.txt 2>&1
#CUDA_VISIBLE_DEVICES=1 python create_embedding_visualization.py splade WTQ > result_splade.txt 2>&1
#CUDA_VISIBLE_DEVICES=1 python create_embedding_visualization.py bge WTQ > result_bge.txt 2>&1