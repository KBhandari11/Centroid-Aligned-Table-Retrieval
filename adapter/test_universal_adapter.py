import os
import sys
import gc
import json
import pickle
import heapq
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from ranx import evaluate, Qrels, Run

from transformers import AutoModel
from sentence_transformers import SentenceTransformer, SparseEncoder

from ..utils.cache_embedding import cache_exists, load_cache_mmap
from ..utils.Adapter import ResidualCentroidAdapter
from ..utils.Adapter import BottleneckResidualAdapter

# (not strictly needed for eval, but imported per your request)
from ..utils.losses import invariance_to_centroid_loss, variance_loss, covariance_loss, identity_preservation_loss
from ..utils.dataset import CacheMultiViewDatasetGiant, collate_fn
from ..utils.TrainerConfig import TrainCfg


# -----------------------------
# your rep list
# -----------------------------
BASE_REPS = [
    'pipe_serialized', 'token_serialized', 'space_serialized',
    "csv", "tsv", "html", "markdown", "latex", "dict", "json", "xml",
    "shuffled_rows", "shuffled_cols", "transpose", "mschema", "macschema", "ddl"
]
CENTROID_REPS = ["centroid_popular", "centroid_data", "centroid_structural", "centroid_schema", "centroid_all"]
SECOND_LEVEL = BASE_REPS + CENTROID_REPS


# -----------------------------
# helpers copied from your pipeline
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

def batched_topk(doc_embeddings, q, k=100, batch_size=8192):
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

    heap.sort(key=lambda x: x[0], reverse=True)
    top_scores = np.array([x[0] for x in heap], dtype=np.float32)
    top_idx = np.array([x[1] for x in heap], dtype=np.int64)
    return top_idx, top_scores


# -----------------------------
# Adapter IO + application
# -----------------------------
def load_adapter(adapter_path: str, device: str = "cuda") -> Tuple[BottleneckResidualAdapter, int, dict]:
    ckpt = torch.load(adapter_path, map_location="cpu")
    D = int(ckpt["D"])
    cfg = ckpt.get("cfg", {})
    '''adapter = BottleneckResidualAdapter(
        d=D,
        hidden_mult=cfg.get("hidden_mult", 4),
        alpha=cfg.get("alpha", 0.1),
        dropout=cfg.get("dropout", 0.0),
    )'''
    adapter = BottleneckResidualAdapter(
        d=D,
        r=cfg.get("r", None),
        alpha=cfg.get("alpha", None),
        dropout=cfg.get("dropout", None),
        use_bias=cfg.get("use_bias", None)
    ) 
    adapter.load_state_dict(ckpt["state_dict"], strict=True)
    adapter.eval().to(device)
    return adapter, D, ckpt

@torch.no_grad()
def adapt_query(adapter: BottleneckResidualAdapter, q: np.ndarray, device="cuda", normalize=True) -> np.ndarray:
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
    adapted = np.zeros( shape=(N, D), dtype=np.float32)

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
# Main evaluation
# -----------------------------
def retrieval_with_adapter(
    model_name: str,
    model,
    all_questions_dict: dict,
    all_table_corpus: dict,
    accelerator,
    adapter_path: Optional[str],
    model_train_date: str, 
    result_save_file: str
):
    dataset_list = ["WTQ", "WIKISQL", "NQ"]  # match your current setting
    cache_root = "/data/Kushal/UniversalRetrieval_data/emb_cache"

    adapter = None
    adapter_dim = None

    if adapter_path is not None and model_name != "bm25":
        adapter, adapter_dim, _ = load_adapter(adapter_path, device="cuda")
        print(f"[adapter] loaded {adapter_path} (D={adapter_dim})", flush=True, file=sys.stderr)

    for dataset in dataset_list:
        all_results = {}
        questions_dict = all_questions_dict[dataset]

        print(dataset, flush=True, file=sys.stderr)

        if not accelerator.is_main_process:
            accelerator.wait_for_everyone()
            continue

        # Evaluate each rep / centroid cache
        for rep_name in SECOND_LEVEL:
            print("\t perturbation", rep_name, flush=True, file=sys.stderr)

            if model_name == "bm25":
                raise NotImplementedError("bm25 path omitted here; your original code can be reused.")
            else:
                if not cache_exists(cache_root, model_name, dataset, rep_name):
                    print(f"Skipped:{rep_name} (no cache)", flush=True)
                    continue

                doc_ids, doc_embeddings = load_cache_mmap(cache_root, model_name, dataset, rep_name)
                if len(doc_ids) == 0:
                    print(f"Skipped:{rep_name} (no docs)", flush=True)
                    continue

                if adapter is not None:
                    if doc_embeddings.shape[1] != adapter_dim:
                        raise ValueError(
                            f"Adapter dim {adapter_dim} != doc dim {doc_embeddings.shape[1]} for {rep_name}"
                        )

                    doc_embeddings = build_adapted_doc_memmap(
                        adapter=adapter,
                        doc_embeddings=doc_embeddings,
                        device="cuda",
                        normalize=True,
                        batch_size=8192,
                    )

                doc_id_set = set(doc_ids)

            qrels_dict = {}
            runs_dict = {}
            num_questions_processed = 0

            for question_id, qd in questions_dict.items():
                question = str(qd["question"])
                gold_table_id = str(qd["gold_table_id"])

                if gold_table_id not in doc_id_set:
                    continue
                num_questions_processed += 1

                q = encode_query(model_name, model, question)

                #if adapter is not None:
                #    q = adapt_query(adapter, q, device="cuda", normalize=True)

                K = 100
                top_idx, top_scores = batched_topk(doc_embeddings, q, k=K, batch_size=8192)
                ranked_doc_ids = [doc_ids[i] for i in top_idx]
                ranked_scores = [float(s) for s in top_scores]

                qid = str(question_id)
                qrels_dict[qid] = {gold_table_id: 1}
                runs_dict[qid] = {str(did): score for did, score in zip(ranked_doc_ids, ranked_scores)}

            if num_questions_processed == 0:
                print(f"Skipped:{rep_name} (no questions)", flush=True)
                doc_embeddings = None
                doc_ids = None
                doc_id_set = None
                gc.collect()
                continue

            qrels = Qrels(qrels_dict)
            runs = Run(runs_dict)
            metrics = evaluate(
                qrels, runs,
                ["recall@1", "recall@5", "recall@10", "recall@50", "recall@100", "recall@500",
                 "ndcg@1", "ndcg@5", "ndcg@10", "ndcg@50"],
                make_comparable=True
            )

            all_results[str(rep_name)] = {
                "recall@1": metrics["recall@1"],
                "recall@5": metrics["recall@5"],
                "recall@10": metrics["recall@10"],
                "recall@50": metrics["recall@50"],
                "recall@100": metrics["recall@100"],
                "recall@500": metrics["recall@500"],
                "ndcg@1": metrics["ndcg@1"],
                "ndcg@5": metrics["ndcg@5"],
                "ndcg@10": metrics["ndcg@10"],
                "ndcg@50": metrics["ndcg@50"],
                "num_questions": num_questions_processed,
                "num_tables": len(doc_ids),
            }

            print(f"\t\tCompleted: {rep_name}", flush=True)
            print(f"\t\tQuestions: {num_questions_processed}, Tables: {len(doc_ids)}", flush=True)

            doc_embeddings = None
            doc_ids = None
            doc_id_set = None
            gc.collect()

        # Save results
        save_dir = f"/data/Kushal/UniversalRetrieval_data/retrieval_all/{model_name}_results/{dataset}/{model_train_date}/"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, result_save_file) #perturbation_results_with_adapter_subset if subset 
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved results to: {save_path}\n")

        table_corpus = None
        gc.collect()


if __name__ == "__main__":
    model_name = sys.argv[1]          # e.g., reasonir / bge / mpnet / jina / splade
    model_train_date = sys.argv[2]  # 2026-03-01
    subset  = sys.argv[3]

    accelerator = Accelerator()
    model = get_model(accelerator, model_name=model_name)
    # if subset: centroid_adapter_subset_dataset 
    if subset == "subset":
        adapter_path = f"/data/Kushal/UniversalRetrieval_data/{model_train_date}/centroid_adapter_subset_dataset/{model_name}/adapter.pt" 
        result_save_file  =  "perturbation_results_with_adapter_subset.json" 
    else:
        adapter_path = f"/data/Kushal/UniversalRetrieval_data/{model_train_date}/centroid_adapter/{model_name}/adapter.pt" 
        result_save_file  = "perturbation_results_with_adapter.json" 

       

    all_questions_dict = get_data('/data/Kushal/UniversalRetrieval_data/dataset_all/new_all_questions_dict_wtq_nqt_wikisql.pkl')
    table_corpus = get_table_corpus("/data/Kushal/UniversalRetrieval_data/dataset_all/all_table_transform_table_corpus.pkl")


    retrieval_with_adapter(model_name, model, all_questions_dict, table_corpus, accelerator, adapter_path, model_train_date, result_save_file)