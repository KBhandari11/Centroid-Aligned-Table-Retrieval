import os
import sys
import gc
import json
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import AutoModel
from sentence_transformers import SentenceTransformer, SparseEncoder

from ..utils.cache_embedding import cache_exists, load_cache_mmap
from ..utils.Adapter import BottleneckResidualAdapter


# -----------------------------
# rep lists
# -----------------------------
BASE_REPS = [
    "pipe_serialized", "token_serialized", "space_serialized",
    "csv", "tsv", "html", "markdown", "latex", "dict", "json", "xml",
    "shuffled_rows", "shuffled_cols", "transpose",
    "mschema", "macschema", "ddl",
]

CENTROID_REPS = [
    "centroid_popular", "centroid_data",
    "centroid_structural", "centroid_schema", "centroid_all"
]

ALL_REPS = BASE_REPS + CENTROID_REPS


# -----------------------------
# helpers
# -----------------------------
def get_data(filename: str) -> dict:
    with open(filename, "rb") as f:
        return pickle.load(f)

def get_table_corpus(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

def get_model(accelerator, model_name="reasonir"):
    if "splade" in model_name:
        return SparseEncoder("naver/splade-v3")

    if "reasonir" in model_name:
        model = AutoModel.from_pretrained(
            "reasonir/ReasonIR-8B",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        model.eval()
        return model

    if "rank1" in model_name:
        model = AutoModel.from_pretrained(
            "jhu-clsp/rank1-7b",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        model.eval()
        return model

    if "bge" in model_name:
        return SentenceTransformer(
            "BAAI/bge-m3",
            model_kwargs={"dtype": torch.float16, "device_map": "auto"},
            device="cuda",
        )

    if "mpnet" in model_name:
        return SentenceTransformer("all-mpnet-base-v2", device="cuda")

    if "jina" in model_name:
        return SentenceTransformer(
            "jinaai/jina-embeddings-v3",
            model_kwargs={"dtype": torch.float16, "device_map": "auto"},
            trust_remote_code=True,
        )

    if "bm25" in model_name:
        return None

    raise ValueError(f"Unknown model_name={model_name}")


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


def batched_dot_scores(doc_embeddings, q, batch_size=8192):
    n = doc_embeddings.shape[0]
    q = np.ascontiguousarray(q, dtype=np.float32)
    scores = np.empty(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        scores[start:end] = (doc_embeddings[start:end] @ q).astype(np.float32, copy=False)

    return scores


# -----------------------------
# Adapter IO + application
# -----------------------------
def load_adapter(adapter_path: str, device: str = "cuda") -> Tuple[BottleneckResidualAdapter, int, dict]:
    ckpt = torch.load(adapter_path, map_location="cpu")
    D = int(ckpt["D"])
    cfg = ckpt.get("cfg", {})

    adapter = BottleneckResidualAdapter(
        d=D,
        r=cfg.get("r", None),
        alpha=cfg.get("alpha", None),
        dropout=cfg.get("dropout", None),
        use_bias=cfg.get("use_bias", None),
    )
    adapter.load_state_dict(ckpt["state_dict"], strict=True)
    adapter.eval().to(device)
    return adapter, D, ckpt


@torch.no_grad()
def adapt_query(
    adapter: BottleneckResidualAdapter,
    q: np.ndarray,
    device="cuda",
    normalize=True
) -> np.ndarray:
    t = torch.from_numpy(np.asarray(q, dtype=np.float32)).to(device)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    z = adapter(t)
    if normalize:
        z = F.normalize(z, p=2, dim=-1)
    z = z.detach().cpu().numpy().astype(np.float32, copy=False)
    return z[0]


def build_adapted_doc_memmap(
    adapter: BottleneckResidualAdapter,
    doc_embeddings,
    device="cuda",
    normalize=True,
    batch_size=8192,
):
    N, D = doc_embeddings.shape
    adapted = np.zeros((N, D), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            block = np.asarray(doc_embeddings[start:end], dtype=np.float32)
            t = torch.from_numpy(block).to(device)
            z = adapter(t)
            if normalize:
                z = F.normalize(z, p=2, dim=-1)
            adapted[start:end] = z.detach().cpu().numpy().astype(np.float32, copy=False)

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

    if adapter_path is not None and model_name != "bm25":
        adapter, adapter_dim, _ = load_adapter(adapter_path, device="cuda")
        print(f"[adapter] loaded {adapter_path} (D={adapter_dim})", flush=True)

    for dataset in dataset_list:
        print(f"\n[Dataset] {dataset}", flush=True)

        questions_dict = all_questions_dict[dataset]

        if not accelerator.is_main_process:
            accelerator.wait_for_everyone()
            continue

        dataset_dir = os.path.join(output_root, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        for perturbation in ALL_REPS:
            print(f"  [Rep] {perturbation}", flush=True)

            if model_name == "bm25":
                raise NotImplementedError("bm25 path is not implemented for rank export with adapter.")

            if not cache_exists(cache_root, model_name, dataset, perturbation):
                print(f"    skipped (no cache): {perturbation}", flush=True)
                continue

            doc_ids, doc_embeddings = load_cache_mmap(cache_root, model_name, dataset, perturbation)

            if len(doc_ids) == 0:
                print(f"    skipped (empty doc list): {perturbation}", flush=True)
                continue

            if adapter is not None:
                if doc_embeddings.shape[1] != adapter_dim:
                    raise ValueError(
                        f"Adapter dim {adapter_dim} != doc dim {doc_embeddings.shape[1]} for {perturbation}"
                    )

                doc_embeddings = build_adapted_doc_memmap(
                    adapter=adapter,
                    doc_embeddings=doc_embeddings,
                    device="cuda",
                    normalize=True,
                    batch_size=8192,
                )

            doc_index = {str(d): i for i, d in enumerate(doc_ids)}

            rep_rows = []
            for qid, qdata in questions_dict.items():
                question = str(qdata["question"])
                gold_table_id = str(qdata["gold_table_id"])

                if gold_table_id not in doc_index:
                    continue

                q = encode_query(model_name, model, question)

                #if adapter is not None and adapt_queries:
                #    q = adapt_query(adapter, q, device="cuda", normalize=True)

                scores = batched_dot_scores(doc_embeddings, q, batch_size=8192)
                gold_idx = doc_index[gold_table_id]
                gold_score = float(scores[gold_idx])

                rank = int(np.sum(scores > gold_score) + 1)

                rep_rows.append({
                    "model": model_name,
                    "dataset": dataset,
                    "perturbation": perturbation,
                    "question_id": str(qid),
                    "gold_table_id": gold_table_id,
                    "rank": rank,
                    "gold_score": gold_score,
                    "num_docs": len(doc_ids),
                    "hit@1": int(rank <= 1),
                    "hit@5": int(rank <= 5),
                    "hit@10": int(rank <= 10),
                    "hit@50": int(rank <= 50),
                    "hit@100": int(rank <= 100),
                })

            if len(rep_rows) == 0:
                print(f"    skipped (no usable questions): {perturbation}", flush=True)
                doc_embeddings = None
                doc_ids = None
                gc.collect()
                continue

            rep_df = pd.DataFrame(rep_rows)

            doc_embeddings = None
            doc_ids = None
            gc.collect()

            rep_csv = os.path.join(dataset_dir, f"{perturbation}_gold_rank_per_question.csv")
            rep_df.to_csv(rep_csv, index=False)
            print(f"    saved: {rep_csv}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    # --------------------------------------------------
    # Args
    # --------------------------------------------------
    model_name = sys.argv[1]         # reasonir / rank1 / bge / mpnet / jina / splade
    model_train_date = sys.argv[2]   # e.g. 2026-03-01
    subset = sys.argv[3]             # "subset" or anything else

    accelerator = Accelerator()

    dataset_list = ["WTQ", "WIKISQL", "NQ"]
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
        adapt_queries=False,  # keep aligned with current test_universal_adapter.py
    )