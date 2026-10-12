import os
import sys
import gc

import numpy as np
import pandas as pd
import scipy.sparse
import torch
from accelerate import Accelerator
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cache_embedding import cache_exists_for_model, load_cache_for_model
from utils.core import (
    SPLADE_TOPK,
    _topk_dense_to_sparse,
    get_data,
    get_table_corpus,
    score_top3_chunk_mean,
    FORMATS_17,
)
from utils.model import get_model
from utils.Adapter import HierDenseAdapter, HierSparseAdapter, FMT2IDX


# -----------------------------
# rep lists (chunked)
# -----------------------------
CATEGORY_MAP = {
    "centroid_popular":    ["pipe_serialized", "token_serialized", "space_serialized"],
    "centroid_data":       ["csv", "tsv", "html", "markdown", "latex", "dict", "json", "xml"],
    "centroid_structural": ["shuffled_rows", "shuffled_cols", "transpose"],
    "centroid_schema":     ["mschema", "macschema", "ddl"],
}

CHUNKED_BASE_REPS    = [f"chunked_{f}" for f in FORMATS_17]
CHUNKED_CENTROID_REPS = (
    [f"chunked_centroid_{k}" for k in CATEGORY_MAP]
    + ["chunked_centroid_all"]
)
ALL_CHUNKED_REPS = CHUNKED_BASE_REPS + CHUNKED_CENTROID_REPS


# -----------------------------
# Query encoding
# -----------------------------
def encode_query(model_name: str, model, question: str) -> np.ndarray:
    if model_name == "reasonir":
        q = model.encode([question], instruction="")
    elif model_name == "jina":
        q = model.encode([question], task="retrieval.query", prompt_name="retrieval.query")
    elif model_name == "splade":
        q = model.encode_query([question]).to_dense()
    else:
        q = model.encode([question])

    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().float().numpy()
    else:
        q = np.asarray(q, dtype=np.float32)

    return q.squeeze(0).astype(np.float32, copy=False)


# -----------------------------
# Adapter IO + application (hierarchical adapters, dot-product space)
# -----------------------------
def load_adapter(adapter_path: str, device: str = "cuda"):
    ckpt = torch.load(adapter_path, map_location="cpu", weights_only=False)
    D = int(ckpt["D"])
    cfg = ckpt.get("cfg", {})
    is_sparse = (ckpt.get("model_name") == "splade")

    if is_sparse:
        adapter = HierSparseAdapter(
            D, r=cfg.get("r", 256), dropout=cfg.get("dropout", 0.05),
            k_exp=cfg.get("k_exp", 0), topk=SPLADE_TOPK,
        )
    else:
        adapter = HierDenseAdapter(D, r=cfg.get("r", 256), dropout=cfg.get("dropout", 0.05))

    adapter.load_state_dict(ckpt["state_dict"], strict=True)
    adapter.eval().to(device)
    return adapter, D, ckpt, is_sparse


def _fmt_tensor(perturbation: str, n: int, device: str) -> Optional[torch.Tensor]:
    """Base reps map to a known format index. Centroid reps use the format head (None).
    Strips the 'chunked_' prefix before looking up in FMT2IDX."""
    base = perturbation.removeprefix("chunked_")
    if base in FMT2IDX:
        return torch.full((n,), FMT2IDX[base], dtype=torch.long, device=device)
    return None


def build_adapted_docs(
    adapter,
    doc_embeddings,
    perturbation: str,
    is_sparse: bool,
    device: str = "cuda",
    batch_size: int = 8192,
):
    """
    Returns raw adapted document embeddings in the same space the adapter was
    trained in. No normalisation: retrieval is dot product on both base and
    adapted sides. Dense -> float32 ndarray. Sparse -> CSR with top-K rows.
    """
    N, D = doc_embeddings.shape
    outs = []
    adapted = None if is_sparse else np.zeros((N, D), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            block = doc_embeddings[start:end]
            block = block.toarray() if scipy.sparse.issparse(block) else np.asarray(block)
            t = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32)).to(device)
            fmt = _fmt_tensor(perturbation, t.shape[0], device)
            z = adapter(t, fmt)[0].detach().cpu().numpy().astype(np.float32, copy=False)
            if is_sparse:
                outs.append(_topk_dense_to_sparse(z, SPLADE_TOPK))
            else:
                adapted[start:end] = z

    if is_sparse:
        return scipy.sparse.vstack(outs, format="csr")
    return adapted


# -----------------------------
# Rank export
# -----------------------------
def export_gold_ranks_with_adapter(
    model_name: str,
    model,
    dataset_list: list,
    all_questions_dict: dict,
    all_table_corpus: dict,
    accelerator,
    cache_root: str,
    output_root: str,
    adapter_path: Optional[str] = None,
    adapt_queries: bool = False,
):
    adapter = None
    adapter_dim = None
    is_sparse = (model_name == "splade")

    if adapter_path is not None and model_name != "bm25":
        adapter, adapter_dim, ckpt, is_sparse_ckpt = load_adapter(adapter_path, device="cuda")
        if ckpt.get("scoring", "dot_product") != "dot_product":
            raise ValueError("Adapter checkpoint was not trained for dot-product scoring.")
        if is_sparse_ckpt != is_sparse:
            raise ValueError("Adapter sparsity does not match model_name.")
        print(f"[adapter] loaded {adapter_path} (D={adapter_dim}, sparse={is_sparse})", flush=True)

    for dataset in dataset_list:
        print(f"\n[Dataset] {dataset}", flush=True)

        questions_dict = all_questions_dict[dataset]

        if not accelerator.is_main_process:
            accelerator.wait_for_everyone()
            continue

        dataset_dir = os.path.join(output_root, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        for rep_key in ALL_CHUNKED_REPS:
            rep_csv = os.path.join(dataset_dir, f"{rep_key}_gold_rank_per_question.csv")
            if os.path.exists(rep_csv):
                print(f"  [Rep] {rep_key}: already scored, skipping", flush=True)
                continue

            print(f"  [Rep] {rep_key}", flush=True)

            if model_name == "bm25":
                raise NotImplementedError("bm25 path is not implemented for rank export with adapter.")

            if not cache_exists_for_model(cache_root, model_name, dataset, rep_key):
                print(f"    skipped (no cache): {rep_key}", flush=True)
                continue

            doc_ids, doc_embeddings = load_cache_for_model(cache_root, model_name, dataset, rep_key)

            if len(doc_ids) == 0:
                print(f"    skipped (empty doc list): {rep_key}", flush=True)
                continue

            if adapter is not None:
                if doc_embeddings.shape[1] != adapter_dim:
                    raise ValueError(
                        f"Adapter dim {adapter_dim} != doc dim {doc_embeddings.shape[1]} for {rep_key}"
                    )
                doc_embeddings = build_adapted_docs(
                    adapter=adapter,
                    doc_embeddings=doc_embeddings,
                    perturbation=rep_key,
                    is_sparse=is_sparse,
                    device="cuda",
                    batch_size=8192,
                )

            rep_rows = []
            for qid, qdata in questions_dict.items():
                question = str(qdata["question"])
                gold_table_id = str(qdata["gold_table_id"])

                q = encode_query(model_name, model, question)

                # score_top3_chunk_mean aggregates chunk scores per table
                ranked = score_top3_chunk_mean(
                    doc_ids, doc_embeddings, q, k=len(doc_ids)
                )
                # ranked = [(table_id, primary_score, max_score), ...]

                primary_scores = {r[0]: r[1] for r in ranked}
                max_scores     = {r[0]: r[2] for r in ranked}

                if gold_table_id not in primary_scores:
                    continue

                gold_primary = primary_scores[gold_table_id]
                gold_max     = max_scores[gold_table_id]

                rank = int(sum(1 for r in ranked if r[1] > gold_primary) + 1)

                rep_rows.append({
                    "model":         model_name,
                    "dataset":       dataset,
                    "format":        rep_key,
                    "question_id":   str(qid),
                    "gold_table_id": gold_table_id,
                    "rank":          rank,
                    "gold_score_top3_mean": gold_primary,
                    "gold_score_max":       gold_max,
                    "num_tables":    len(ranked),
                    "hit@1":   int(rank <= 1),
                    "hit@5":   int(rank <= 5),
                    "hit@10":  int(rank <= 10),
                    "hit@50":  int(rank <= 50),
                    "hit@100": int(rank <= 100),
                })

            doc_embeddings = None
            doc_ids = None
            gc.collect()

            if not rep_rows:
                print(f"    skipped (no usable questions): {rep_key}", flush=True)
                continue

            rep_df = pd.DataFrame(rep_rows)
            rep_df.to_csv(rep_csv, index=False)
            print(f"    saved: {rep_csv}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    model_name = sys.argv[1]         # reasonir / rank1 / bge / mpnet / jina / splade
    model_train_date = sys.argv[2]   # e.g. 2026-03-01
    subset = sys.argv[3]             # "subset" or anything else

    accelerator = Accelerator()

    dataset_list = ["WTQ", "WIKISQL", "NQ"]# if subset == "subset" else ["WTQ", "WIKISQL", "NQ"]
    cache_root = "/data/Kushal/UniversalRetrieval_data/emb_cache"

    questions_path = "/data/Kushal/UniversalRetrieval_data/dataset_all/new_all_questions_dict_wtq_nqt_wikisql.pkl"
    tables_path = "/data/Kushal/UniversalRetrieval_data/dataset_all/all_table_transform_table_corpus.pkl"

    if subset == "subset":
        adapter_path = f"/data/Kushal/UniversalRetrieval_data/{model_train_date}/centroid_adapter_subset_dataset/{model_name}/adapter.pt"
        output_root = f"/data/Kushal/UniversalRetrieval_data/retrieval_all/{model_name}_results_rank_with_adapter_subset/{model_train_date}"
    else:
        adapter_path = f"/data/Kushal/UniversalRetrieval_data/{model_train_date}/centroid_adapter/{model_name}/adapter.pt"
        output_root = f"/data/Kushal/UniversalRetrieval_data/retrieval_all/{model_name}_results_rank_with_adapter/{model_train_date}"

    os.makedirs(output_root, exist_ok=True)

    model = get_model(accelerator, model_name=model_name)
    all_questions_dict = get_data(questions_path)
    all_table_corpus = get_table_corpus(tables_path)

    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        sys.exit(0)

    export_gold_ranks_with_adapter(
        model_name=model_name,
        model=model,
        dataset_list=dataset_list,
        all_questions_dict=all_questions_dict,
        all_table_corpus=all_table_corpus,
        accelerator=accelerator,
        cache_root=cache_root,
        output_root=output_root,
        adapter_path=adapter_path,
        adapt_queries=False,
    )
#CUDA_VISIBLE_DEVICES=1 python -m adapter.test_rank_universal_adapter splade 2026-09-12 subset > ./log/test_splade_subset_rank_chunked.txt 2>&1
#CUDA_VISIBLE_DEVICES=1 python -m adapter.test_rank_universal_adapter bge 2026-09-12 subset > ./log/test_bge_subset_rank_chunked.txt 2>&1
#CUDA_VISIBLE_DEVICES=1 python -m adapter.test_rank_universal_adapter mdbr 2026-09-12 subset > ./log/test_mdbr_subset_rank_chunked.txt 2>&1
#CUDA_VISIBLE_DEVICES=1 python -m adapter.test_rank_universal_adapter miniLM 2026-09-12 subset > ./log/test_miniLM_subset_rank_chunked.txt 2>&1
#CUDA_VISIBLE_DEVICES=1 python -m adapter.test_rank_universal_adapter mpnet 2026-09-12 subset > ./log/test_mpnet_subset_rank_chunked.txt 2>&1
#CUDA_VISIBLE_DEVICES=1 python -m adapter.test_rank_universal_adapter qwen0.6B 2026-09-12 subset > ./log/test_qwen0.6B_subset_rank_chunked.txt 2>&1
#CUDA_VISIBLE_DEVICES=1 python -m adapter.test_rank_universal_adapter reasonir 2026-09-12 subset > ./log/test_reasonir_subset_rank_chunked.txt 2>&1
#CUDA_VISIBLE_DEVICES=1 python -m adapter.test_rank_universal_adapter qwen4B 2026-09-12 subset > ./log/test_qwen4B_subset_rank_chunked.txt 2>&1

#CUDA_VISIBLE_DEVICES=1 python -m adapter.test_rank_universal_adapter splade 2026-09-17 all > ./log/test_splade_rank_chunked_new.txt 2>&1
#CUDA_VISIBLE_DEVICES=1 python -m adapter.test_rank_universal_adapter bge 2026-09-17 all > ./log/test_bge_rank_chunked_new.txt 2>&1
#CUDA_VISIBLE_DEVICES=1 python -m adapter.test_rank_universal_adapter miniLM 2026-09-17 all > ./log/test_miniLM_rank_chunked_new.txt 2>&1
