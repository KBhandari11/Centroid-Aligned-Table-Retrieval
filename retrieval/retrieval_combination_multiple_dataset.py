import gc
import json
import os
import sys

import numpy as np
import torch
from accelerate import Accelerator
from ranx import evaluate, Qrels, Run

# Allow direct execution from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model import get_model
from utils.core import (
    get_data,
    load_manifest,
    score_top3_chunk_mean,
    batched_topk,
    CACHE_ROOT,
    FORMATS_17,
)
from utils.cache_embedding import cache_exists_for_model, load_cache_for_model

QUESTIONS_PATH = "/data/Kushal/UniversalRetrieval_data/dataset_all/new_all_questions_dict_wtq_nqt_wikisql.pkl"

RANX_METRICS = [
    "recall@1", "recall@5", "recall@10", "recall@50", "recall@100", "recall@500",
    "ndcg@1", "ndcg@5", "ndcg@10", "ndcg@50",
]


def combo_key(rep_list):
    return "_".join(rep_list)


# ── Embedding combination helpers ─────────────────────────────────────────────

def compute_combo_embedding(cache_root, model_name, dataset, rep_list,
                            chunked=False, normalize=False):
    """
    Load cached embeddings for each rep in rep_list and mean-combine them.

    For chunked mode (chunked=True):
      - Cache keys are 'chunked_{rep}'
      - doc_ids are '<table_id>::<chunk_id>' strings
      - The manifest guarantees identical doc_id order across all formats,
        so chunk k of rep A aligns with chunk k of rep B by position.
      - Returns (doc_ids, combined_emb) where combined_emb can be passed
        directly to score_top3_chunk_mean.

    For flat mode (chunked=False):
      - Cache keys are the rep names themselves
      - doc_ids are plain table_id strings
      - Returns (doc_ids, combined_emb) for use with batched_topk.
    """
    if not rep_list:
        raise ValueError("rep_list is empty")

    prefix = "chunked_" if chunked else ""
    cache_keys = [f"{prefix}{r}" for r in rep_list]

    anchor_key = cache_keys[0]
    if not cache_exists_for_model(cache_root, model_name, dataset, anchor_key):
        raise ValueError(
            f"Missing cache for anchor '{anchor_key}' in dataset '{dataset}'. "
            f"Run embedding first."
        )

    doc_ids0, emb0 = load_cache_for_model(cache_root, model_name, dataset, anchor_key)

    # ── SPLADE: combine sparse matrices, re-sparsify with top-K ──────────────
    if model_name == "splade":
        from utils.core import _topk_sparse_rows, SPLADE_TOPK
        acc = emb0.astype(np.float32)
        count = 1
        emb0 = None
        for key in cache_keys[1:]:
            if not cache_exists_for_model(cache_root, model_name, dataset, key):
                raise ValueError(
                    f"Missing cache for '{key}' in dataset '{dataset}'. "
                    f"Run embedding first."
                )
            doc_ids_r, emb_r = load_cache_for_model(cache_root, model_name, dataset, key)
            if doc_ids_r != doc_ids0:
                raise ValueError(
                    f"doc_ids mismatch between '{cache_keys[0]}' and '{key}'. "
                    "All reps must be built from the same manifest."
                )
            acc = acc + emb_r.astype(np.float32)
            count += 1
            emb_r = None
            gc.collect()
        combo = _topk_sparse_rows(acc / count, SPLADE_TOPK)
        gc.collect()
        return doc_ids0, combo

    # ── Dense path ────────────────────────────────────────────────────────────
    acc = np.asarray(emb0, dtype=np.float32).copy()
    count = 1
    emb0 = None

    for key in cache_keys[1:]:
        if not cache_exists_for_model(cache_root, model_name, dataset, key):
            raise ValueError(
                f"Missing cache for '{key}' in dataset '{dataset}'. "
                f"Run embedding first."
            )

        doc_ids_r, emb_r = load_cache_for_model(cache_root, model_name, dataset, key)

        # Alignment check: manifest guarantees same order, but verify explicitly
        if doc_ids_r != doc_ids0:
            raise ValueError(
                f"doc_ids mismatch between '{cache_keys[0]}' and '{key}'. "
                "All reps must be built from the same manifest."
            )

        acc += np.asarray(emb_r, dtype=np.float32)
        count += 1
        emb_r = None
        gc.collect()

    combo = (acc / count).astype(np.float32, copy=False)

    if normalize:
        denom = np.linalg.norm(combo, axis=1, keepdims=True) + 1e-12
        combo = combo / denom

    gc.collect()
    return doc_ids0, combo


# ── Query encoder ─────────────────────────────────────────────────────────────

def encode_query(model_name, model, question_text):
    if model_name == "reasonir":
        q = model.encode([question_text], instruction="")
    elif model_name == "jina":
        q = model.encode([question_text], task="retrieval.query",
                         prompt_name="retrieval.query")
    elif model_name == "splade":
        q = model.encode_query([question_text]).to_dense()
    else:
        q = model.encode([question_text])

    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().float().numpy()
    else:
        q = np.asarray(q, dtype=np.float32)
    return q.squeeze(0).astype(np.float32, copy=False)


# ── Main combination function ─────────────────────────────────────────────────

def append_combo_result(
    model_name,
    model,
    dataset,
    rep_list,
    chunked=False,
    cache_root=CACHE_ROOT,
    questions_path=QUESTIONS_PATH,
    manifest_path=None,
):
    """
    Combine embeddings for rep_list, evaluate with ranx, and append/update
    the result in perturbation_results.json (flat) or
    perturbation_results_chunked.json (chunked).

    chunked=True:
      Combines 'chunked_{rep}' caches.  Chunk k of rep A is averaged with
      chunk k of rep B (alignment guaranteed by the shared manifest).
      Scores are collapsed to table level via top-3 chunk mean.

    chunked=False:
      Combines flat (one-per-table) caches.  Backward-compatible with the
      original flat retrieval pipeline.
    """
    all_questions_dict = get_data(questions_path)
    if dataset not in all_questions_dict:
        raise ValueError(f"Dataset '{dataset}' not found in questions file.")
    questions_dict = all_questions_dict[dataset]

    # Build combined embedding
    doc_ids, doc_embeddings = compute_combo_embedding(
        cache_root=cache_root,
        model_name=model_name,
        dataset=dataset,
        rep_list=rep_list,
        chunked=chunked,
        normalize=False,
    )

    if chunked:
        # Load manifest to know which table_ids are valid
        if manifest_path is None:
            manifest_path = os.path.join(
                f"./data/retrieval_all/{model_name}_results_chunked",
                dataset, f"{dataset}_manifest.json",
            )
        manifest = load_manifest(manifest_path)
        valid_table_ids = set(manifest["tables"].keys())
    else:
        doc_id_set = set(doc_ids)

    qrels_dict = {}
    runs_dict  = {}
    num_processed = 0

    for question_id, question_data in questions_dict.items():
        question      = str(question_data["question"])
        gold_table_id = str(question_data["gold_table_id"])

        if chunked:
            if gold_table_id not in valid_table_ids:
                continue
        else:
            if gold_table_id not in doc_id_set:
                continue

        num_processed += 1
        q = encode_query(model_name, model, question)
        qid = str(question_id)

        if chunked:
            # Step 8: top-3 chunk mean → table-level scores
            ranked = score_top3_chunk_mean(
                doc_ids, doc_embeddings, q, k=len(valid_table_ids),
            )
            qrels_dict[qid] = {gold_table_id: 1}
            runs_dict[qid]  = {r[0]: r[1] for r in ranked}
        else:
            K = 100
            top_idx, top_scores = batched_topk(
                doc_embeddings, q, k=K, batch_size=8192,
            )
            qrels_dict[qid] = {gold_table_id: 1}
            runs_dict[qid]  = {
                str(doc_ids[i]): float(s)
                for i, s in zip(top_idx, top_scores)
            }

    doc_embeddings = None
    doc_ids        = None
    gc.collect()

    if num_processed == 0:
        print(f"No valid questions for dataset={dataset}, combo={rep_list}.")
        return

    metrics = evaluate(
        Qrels(qrels_dict), Run(runs_dict),
        RANX_METRICS, make_comparable=True,
    )

    # Save location depends on chunked/flat
    if chunked:
        save_dir  = f"./data/retrieval_all/{model_name}_results_chunked/{dataset}/"
        save_file = "perturbation_results_combo.json"
    else:
        save_dir  = f"./data/retrieval_all/{model_name}_results/{dataset}/"
        save_file = "perturbation_results.json"

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_file)

    all_results = {}
    if os.path.exists(save_path):
        with open(save_path) as f:
            all_results = json.load(f)

    key = ("chunked_" if chunked else "") + combo_key(rep_list)
    all_results[key] = {
        **{m: metrics[m] for m in RANX_METRICS},
        "num_questions": num_processed,
        "combo_reps": rep_list,
        "pooling": "top3_mean" if chunked else "flat",
    }

    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"[DONE] {save_path}  key={key}", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage:
      python retrieval/retrieval_combination_multiple_dataset.py <model> [chunked] <rep1> <rep2> ...

    Examples (flat):
      CUDA_VISIBLE_DEVICES=1 python retrieval/retrieval_combination_multiple_dataset.py mpnet csv tsv pipe_serialized
      CUDA_VISIBLE_DEVICES=5 python retrieval/retrieval_combination_multiple_dataset.py bge xml dict json markdown

    Examples (chunked):
      CUDA_VISIBLE_DEVICES=1 python retrieval/retrieval_combination_multiple_dataset.py mpnet chunked csv tsv pipe_serialized
      CUDA_VISIBLE_DEVICES=5 python retrieval/retrieval_combination_multiple_dataset.py bge chunked xml dict json markdown
    """
    if len(sys.argv) < 3:
        print("Usage: python retrieval_combination_multiple_dataset.py "
              "<model_name> [chunked] <rep1> <rep2> ...")
        sys.exit(1)

    model_name = sys.argv[1]

    if sys.argv[2] == "chunked":
        chunked  = True
        rep_list = sys.argv[3:]
    else:
        chunked  = False
        rep_list = sys.argv[2:]

    if not rep_list:
        print("ERROR: provide at least one representation name.")
        sys.exit(1)

    for rep in rep_list:
        if rep not in FORMATS_17:
            print(f"ERROR: unknown representation '{rep}'.")
            sys.exit(1)

    dataset_list = ["WTQ", "WIKISQL", "NQ"]

    accelerator = Accelerator()
    model = get_model(accelerator, model_name=model_name)

    for dataset in dataset_list:
        print(f"Dataset: {dataset}", flush=True)
        append_combo_result(
            model_name=model_name,
            model=model,
            dataset=dataset,
            rep_list=rep_list,
            chunked=chunked,
            cache_root=CACHE_ROOT,
            questions_path=QUESTIONS_PATH,
        )
