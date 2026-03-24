import os
import sys
import json
import gc
import numpy as np
import torch
from accelerate import Accelerator
from ranx import evaluate, Qrels, Run

# Import existing functions from your original file
from ..utils.model import get_model
from ..utils.core import (
    get_data,
    load_cache_mmap,
    batched_topk,
    CACHE_ROOT,

)
from ..utils.cache_embedding import *

QUESTIONS_PATH = "./data/dataset_all/new_all_questions_dict_wtq_nqt_wikisql.pkl"
REPRESENTATION = ["pipe_serialized", "token_serialized", "space_serialized",
                  "csv", "tsv", "html", "markdown", "latex", "dict", "json", "xml",
                  "shuffled_rows", "shuffled_cols", "transpose",
                  "mschema", "macschema", "ddl"
                  ]

def combo_key(rep_list):
    # Keep order as provided by user
    # Example: ["pipe_serialized","tsv","ddl"] -> "pipe_serialized_tsv_ddl"
    return "_".join(rep_list)


def compute_combo_embedding_from_cache(
    cache_root,
    model_name,
    dataset,
    rep_list,
    normalize=False,
):
    """
    Load cached embeddings for reps and mean-combine in memory.
    Returns:
      doc_ids (list[str]), combo_emb (np.ndarray [N,D] float32)
    """
    if not rep_list:
        raise ValueError("rep_list is empty")

    anchor = rep_list[0]
    if not cache_exists(cache_root, model_name, dataset, anchor):
        raise ValueError(f"Missing cached embedding for anchor rep '{anchor}' in dataset '{dataset}'")

    doc_ids0, emb0 = load_cache_mmap(cache_root, model_name, dataset, anchor)
    N, D = emb0.shape

    acc = np.zeros((N, D), dtype=np.float32)
    count = 0

    for rep in rep_list:
        if not cache_exists(cache_root, model_name, dataset, rep):
            raise ValueError(f"Missing cached embedding for rep '{rep}' in dataset '{dataset}'")

        doc_ids_r, emb_r = load_cache_mmap(cache_root, model_name, dataset, rep)
        if doc_ids_r != doc_ids0:
            raise ValueError(
                f"doc_ids mismatch between '{anchor}' and '{rep}'. "
                "Cached reps are not aligned."
            )

        acc += np.asarray(emb_r, dtype=np.float32)
        count += 1

        emb_r = None
        gc.collect()

    combo = acc / max(count, 1)

    if normalize:
        denom = np.linalg.norm(combo, axis=1, keepdims=True) + 1e-12
        combo = combo / denom

    emb0 = None
    gc.collect()
    return doc_ids0, combo.astype(np.float32, copy=False)


def encode_query(model_name, model, question_text):
    if model_name == "reasonir":
        q = model.encode([question_text], instruction="")
    elif model_name == "jina":
        q = model.encode([question_text], task="retrieval.query", prompt_name="retrieval.query")
    elif model_name == "splade":
        q = model.encode_query([question_text]).to_dense()
    else:
        q = model.encode([question_text])

    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().float().numpy()
    else:
        q = np.asarray(q, dtype=np.float32)

    return q.squeeze(0).astype(np.float32, copy=False)  # (D,)


def append_combo_result(
    model_name,
    dataset,
    rep_list,
    cache_root=CACHE_ROOT,
    questions_path=QUESTIONS_PATH,
):
    """
    Appends/updates one combo result in:
    ./retrieval_all/{model_name}_results/{dataset}/perturbation_results.json
    """
    accelerator = Accelerator()
    model = get_model(accelerator, model_name=model_name)

    # Load questions for all datasets, then select one
    all_questions_dict = get_data(questions_path)
    if dataset not in all_questions_dict:
        raise ValueError(f"Dataset '{dataset}' not found in questions file.")
    questions_dict = all_questions_dict[dataset]

    # Load / build combo embedding from existing caches only
    doc_ids, doc_embeddings = compute_combo_embedding_from_cache(
        cache_root=cache_root,
        model_name=model_name,
        dataset=dataset,
        rep_list=rep_list,
        normalize=False,
    )
    doc_id_set = set(doc_ids)

    qrels_dict = {}
    runs_dict = {}
    num_questions_processed = 0

    for question_id, question_data in questions_dict.items():
        question = str(question_data["question"])
        gold_table_id = str(question_data["gold_table_id"])

        if gold_table_id not in doc_id_set:
            continue

        num_questions_processed += 1
        q = encode_query(model_name, model, question)

        # (N, D) @ (D,) => (N,)
        '''arr_scores = np.asarray(doc_embeddings @ q, dtype=np.float32)

        ranked_idx = np.argsort(arr_scores)[::-1]
        ranked_doc_ids = [doc_ids[i] for i in ranked_idx]
        ranked_scores = [float(arr_scores[i]) for i in ranked_idx]'''
        K = 100
        top_idx, top_scores = batched_topk(doc_embeddings, q, k=K, batch_size=8192)
        ranked_doc_ids = [doc_ids[i] for i in top_idx]
        ranked_scores = [float(s) for s in top_scores]

        qid = str(question_id)
        qrels_dict[qid] = {gold_table_id: 1}
        runs_dict[qid] = {
            str(did): score for did, score in zip(ranked_doc_ids, ranked_scores)
        }

    if num_questions_processed == 0:
        print(f"No valid questions for dataset={dataset}, combo={rep_list}. Nothing written.")
        return

    qrels = Qrels(qrels_dict)
    runs = Run(runs_dict)
    metrics = evaluate(
        qrels, runs,
        [
            "recall@1", "recall@5", "recall@10", "recall@50", "recall@100", "recall@500",
            "ndcg@1", "ndcg@5", "ndcg@10", "ndcg@50"
        ],
        make_comparable=True
    )

    # Load existing perturbation_results.json
    save_dir = f"./data/retrieval_all/{model_name}_results/{dataset}/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "perturbation_results.json")

    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    key = combo_key(rep_list)
    all_results[key] = {
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
        "combo_reps": rep_list,
    }

    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"[DONE] Updated: {save_path}")
    print(f"[DONE] Added/updated key: {key}",flush=True)

    # Cleanup
    doc_embeddings = None
    doc_ids = None
    doc_id_set = None
    gc.collect()


if __name__ == "__main__":
    """
    Example:
      CUDA_VISIBLE_DEVICES=5 python append_combo_result.py reasonir WTQ pipe_serialized tsv
      CUDA_VISIBLE_DEVICES=5 python append_combo_result.py bge NQ json mschema ddl
    """
    if len(sys.argv) < 4:
        print("Usage: python append_combo_result.py <model_name> <rep1> <rep2> [rep3 ...]")
        sys.exit(1)

    

    model_name = sys.argv[1]
    rep_list = sys.argv[2:]
    dataset_list = ["WTQ","WIKISQL","NQ"]#["WTQ","WIKISQL"]
    #dataset_list = ["NQ"]#["WTQ","WIKISQL"]


    for rep in rep_list:
        if rep not in REPRESENTATION:
            print("ERROR: Representation is incorrect!")
            sys.exit(1)
        
    for dataset in dataset_list:
        print("Dataset:", dataset, flush=True)
        append_combo_result(
            model_name=model_name,
            dataset=dataset,
            rep_list=rep_list,
            cache_root=CACHE_ROOT,
            questions_path=QUESTIONS_PATH,
        )


    #CUDA_VISIBLE_DEVICES=1 python retrieval_combination_multiple_dataset.py mpnet tsv csv pipe_serialized space_serialized transpose > result_mpnet.txt 2>&1
    #CUDA_VISIBLE_DEVICES=1 python retrieval_combination_multiple_dataset.py splade tsv csv space_serialized ddl latex > result_splade.txt 2>&1
    # CUDA_VISIBLE_DEVICES=2,3 python retrieval_combination_multiple_dataset.py reasonir xml ddl latex html token_serialized > result_reasonir.txt 2>&1
    # CUDA_VISIBLE_DEVICES=5 python retrieval_combination_multiple_dataset.py bge xml dict json pipe_serialized markdown > result_bge.txt 2>&1

    # CUDA_VISIBLE_DEVICES=1 python retrieval_combination_multiple_dataset.py mpnet csv tsv pipe_serialized space_serialized xml latex> result_mpnet.txt 2>&1
    # CUDA_VISIBLE_DEVICES=1 python retrieval_combination_multiple_dataset.py splade csv tsv pipe_serialized space_serialized xml latex > result_splade.txt 2>&1
    # CUDA_VISIBLE_DEVICES=2,3 python retrieval_combination_multiple_dataset.py reasonir csv tsv pipe_serialized space_serialized xml latex > result_reasonir.txt 2>&1
    # CUDA_VISIBLE_DEVICES=5 python retrieval_combination_multiple_dataset.py bge csv tsv pipe_serialized space_serialized xml latex> result_bge.txt 2>&1

