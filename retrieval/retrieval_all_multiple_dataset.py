import gc
import os
import re
import sys

import numpy as np
import torch
from accelerate import Accelerator
from ranx import evaluate, Qrels, Run

# Allow direct execution: add the repo root (parent of this file's directory)
# to sys.path so absolute imports work regardless of working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model import get_model
from utils.core import (
    get_data,
    get_table_corpus,
    compute_and_cache_rep_embeddings,
    compute_and_cache_centroid,
    compute_and_cache_chunked_embeddings,
    build_manifest,
    load_manifest,
    get_tokenizer_from_model,
    score_top3_chunk_mean,
    sanity_check_chunk_alignment,
    batched_topk,
    CACHE_ROOT,
    FORMATS_17,
)
from utils.cache_embedding import cache_exists_for_model, load_cache_for_model

# ── Format lists ──────────────────────────────────────────────────────────────

base_reps = [
    "pipe_serialized", "token_serialized", "space_serialized",
    "csv", "tsv", "html", "markdown", "latex", "dict", "json", "xml",
    "shuffled_rows", "shuffled_cols", "transpose",
    "mschema", "macschema", "ddl",
]
centroid_reps = [
    "centroid_popular", "centroid_data", "centroid_structural",
    "centroid_schema", "centroid_all",
]

category_map = {
    "centroid_popular":    ["pipe_serialized", "token_serialized", "space_serialized"],
    "centroid_data":       ["csv", "tsv", "html", "markdown", "latex", "dict", "json", "xml"],
    "centroid_structural": ["shuffled_rows", "shuffled_cols", "transpose"],
    "centroid_schema":     ["mschema", "macschema", "ddl"],
}

# Step 10: NQ-Tables uses only this subset (selected greedily for complementarity)
NQ_FORMATS = ["csv", "markdown", "token_serialized", "ddl"]

RANX_METRICS = [
    "recall@1", "recall@5", "recall@10", "recall@50", "recall@100", "recall@500",
    "ndcg@1", "ndcg@5", "ndcg@10", "ndcg@50",
]


# ── Query encoder helper ──────────────────────────────────────────────────────

def _encode_query(model, model_name, question):
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


# ── Chunked retrieval (Steps 7-10) ────────────────────────────────────────────

def retrieval_chunked(model_name, model, all_questions_dict, all_table_corpus,
                      accelerator, dataset_list=None, run_sanity_check=True):
    """
    Steps 7-10: encode chunks, score with top-3 chunk mean, evaluate with ranx.

    Results are saved to:
      ./data/retrieval_all/<model>_results_chunked/<dataset>/perturbation_results.json

    Both top-3 mean (primary) and max-pooling (secondary) scores are reported
    so they can be compared directly.

    The tokenizer is extracted directly from the loaded model via
    get_tokenizer_from_model — no separate model name list needed.
    """
    if dataset_list is None:
        dataset_list = ["WTQ", "WIKISQL", "NQ"]

    tokenizers = get_tokenizer_from_model(model, model_name)

    for dataset in dataset_list:
        all_results_top3 = {}
        all_results_max  = {}

        table_corpus    = all_table_corpus[dataset]
        questions_dict  = all_questions_dict[dataset]

        print(dataset, flush=True, file=sys.stderr)

        if not accelerator.is_main_process:
            accelerator.wait_for_everyone()
            continue

        # ── Step 4: build or load manifest ───────────────────────────────────
        manifest_dir  = f"./data/retrieval_all/{model_name}_results_chunked/{dataset}"
        manifest_path = os.path.join(manifest_dir, f"{dataset}_manifest.json")

        if os.path.exists(manifest_path):
            print(f"  Loading manifest: {manifest_path}", flush=True, file=sys.stderr)
            manifest = load_manifest(manifest_path)
        else:
            print(f"  Building manifest for {dataset} …", flush=True, file=sys.stderr)
            manifest = build_manifest(
                table_corpus=table_corpus,
                dataset=dataset,
                tokenizers=tokenizers,
                manifest_path=manifest_path,
            )

        # ── Sanity check (before any encoding) ───────────────────────────────
        if run_sanity_check:
            ok = sanity_check_chunk_alignment(table_corpus, manifest, n_tables=10)
            if not ok:
                print(
                    f"[WARNING] [{dataset}] Sanity check found rendering "
                    "mismatches (likely XML tag sanitization edge cases). "
                    "Chunk boundaries are correct; proceeding.",
                    flush=True,
                )

        # Step 10: restrict formats for NQ
        formats = FORMATS_17 #NQ_FORMATS if dataset == "NQ" else FORMATS_17

        # ── Step 7: ensure chunked embedding caches ───────────────────────────
        for fmt in formats:
            rep_key = f"chunked_{fmt}"
            if not cache_exists_for_model(CACHE_ROOT, model_name, dataset, rep_key):
                print(f"  [cache miss] building {rep_key}", flush=True, file=sys.stderr)
                compute_and_cache_chunked_embeddings(
                    model_name=model_name,
                    model=model,
                    table_corpus=table_corpus,
                    manifest=manifest,
                    format_name=fmt,
                    cache_root=CACHE_ROOT,
                    dataset=dataset,
                    batch_size=256,
                    normalize=False,
                    instruction="",
                )
                gc.collect()

        accelerator.wait_for_everyone()

        # ── Step 8: evaluate each format ─────────────────────────────────────
        for fmt in formats:
            rep_key = f"chunked_{fmt}"
            print(f"\t {rep_key}", flush=True, file=sys.stderr)

            if not cache_exists_for_model(CACHE_ROOT, model_name, dataset, rep_key):
                print(f"  Skipped: {rep_key} (no cache)", flush=True)
                continue

            doc_ids, doc_embeddings = load_cache_for_model(
                CACHE_ROOT, model_name, dataset, rep_key
            )
            if len(doc_ids) == 0:
                print(f"  Skipped: {rep_key} (empty)", flush=True)
                continue

            qrels_dict_top3 = {}
            runs_dict_top3  = {}
            qrels_dict_max  = {}
            runs_dict_max   = {}
            num_processed   = 0

            for question_id, question_data in questions_dict.items():
                question      = str(question_data["question"])
                gold_table_id = str(question_data["gold_table_id"])

                if gold_table_id not in manifest["tables"]:
                    continue

                num_processed += 1
                q = _encode_query(model, model_name, question)

                # Step 8: top-3 chunk mean → ranked table list
                ranked = score_top3_chunk_mean(
                    doc_ids, doc_embeddings, q,
                    k=len(manifest["tables"]),
                )
                # ranked = [(table_id, primary_top3_mean, max_score), ...]

                qid = str(question_id)
                qrels_dict_top3[qid] = {gold_table_id: 1}
                qrels_dict_max[qid]  = {gold_table_id: 1}

                runs_dict_top3[qid] = {r[0]: r[1] for r in ranked}
                runs_dict_max[qid]  = {r[0]: r[2] for r in ranked}

            doc_embeddings = None
            doc_ids        = None
            gc.collect()

            if num_processed == 0:
                print(f"  Skipped: {rep_key} (no questions)", flush=True)
                continue

            # Evaluate top-3 mean (primary)
            metrics_top3 = evaluate(
                Qrels(qrels_dict_top3), Run(runs_dict_top3),
                RANX_METRICS, make_comparable=True,
            )
            # Evaluate max pooling (secondary)
            metrics_max = evaluate(
                Qrels(qrels_dict_max), Run(runs_dict_max),
                RANX_METRICS, make_comparable=True,
            )

            all_results_top3[rep_key] = {
                **{m: metrics_top3[m] for m in RANX_METRICS},
                "num_questions": num_processed,
                "pooling": "top3_mean",
            }
            all_results_max[rep_key] = {
                **{m: metrics_max[m] for m in RANX_METRICS},
                "num_questions": num_processed,
                "pooling": "max",
            }

            print(
                f"\t\tCompleted: {rep_key}  questions={num_processed}  "
                f"recall@10(top3)={metrics_top3['recall@10']:.4f}  "
                f"recall@10(max)={metrics_max['recall@10']:.4f}",
                flush=True,
            )

        # Save results
        save_dir = os.path.join(
            f"./data/retrieval_all/{model_name}_results_chunked", dataset
        )
        os.makedirs(save_dir, exist_ok=True)

        for suffix, results in [("top3_mean", all_results_top3), ("max", all_results_max)]:
            save_path = os.path.join(save_dir, f"perturbation_results_{suffix}.json")
            import json
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved {suffix} results → {save_path}\n")

        table_corpus = None
        gc.collect()


# ── Legacy flat retrieval (unchanged) ─────────────────────────────────────────

def bm25_tokenize(text):
    if not isinstance(text, str):
        text = str(text)
    return re.findall(r"[A-Za-z0-9_]+|[|,]", text.lower())


def retrieval(model_name, model, all_questions_dict, all_table_corpus, accelerator):
    dataset_list = ["WTQ", "WIKISQL"]

    second_level = base_reps + centroid_reps

    for dataset in dataset_list:
        all_results = {}
        table_corpus = all_table_corpus[dataset]
        questions_dict = all_questions_dict[dataset]
        all_table_ids_list = list(table_corpus.keys())

        print(dataset, flush=True, file=sys.stderr)

        if not accelerator.is_main_process:
            accelerator.wait_for_everyone()
            continue

        if model_name != "bm25":
            for rep in base_reps:
                if not cache_exists_for_model(CACHE_ROOT, model_name, dataset, rep):
                    print(f"[cache miss] building {rep}", flush=True, file=sys.stderr)
                    compute_and_cache_rep_embeddings(
                        model_name=model_name,
                        model=model,
                        table_corpus=table_corpus,
                        table_ids=all_table_ids_list,
                        rep_name=rep,
                        cache_root=CACHE_ROOT,
                        dataset=dataset,
                        batch_size=256,
                        instruction="",
                        normalize=False,
                    )
                    gc.collect()

            for cent_name, reps in category_map.items():
                if not cache_exists_for_model(CACHE_ROOT, model_name, dataset, cent_name):
                    print(f"[cache miss] building {cent_name}", flush=True, file=sys.stderr)
                    compute_and_cache_centroid(
                        CACHE_ROOT, model_name, dataset, cent_name, reps
                    )
                    gc.collect()

            if not cache_exists_for_model(CACHE_ROOT, model_name, dataset, "centroid_all"):
                print("[cache miss] building centroid_all", flush=True, file=sys.stderr)
                compute_and_cache_centroid(
                    CACHE_ROOT, model_name, dataset, "centroid_all", base_reps
                )
                gc.collect()

        accelerator.wait_for_everyone()

        for perturbation in second_level:
            print(f"\t perturbation {perturbation}", flush=True, file=sys.stderr)

            if model_name == "bm25":
                docs, doc_ids = [], []
                for table_id in all_table_ids_list:
                    if table_id in table_corpus and perturbation in table_corpus[table_id]:
                        docs.append(table_corpus[table_id][perturbation])
                        doc_ids.append(table_id)
                if len(doc_ids) == 0:
                    print(f"Skipped:{perturbation} (no docs)", flush=True)
                    continue
            else:
                if not cache_exists_for_model(CACHE_ROOT, model_name, dataset, perturbation):
                    print(f"Skipped:{perturbation} (no cache)", flush=True)
                    continue

                doc_ids, doc_embeddings = load_cache_mmap(
                    CACHE_ROOT, model_name, dataset, perturbation
                )
                if len(doc_ids) == 0:
                    print(f"Skipped:{perturbation} (no docs)", flush=True)
                    continue

                doc_id_set = set(doc_ids)

            qrels_dict = {}
            runs_dict  = {}
            num_processed = 0

            for question_id, question_data in questions_dict.items():
                question      = str(question_data["question"])
                gold_table_id = str(question_data["gold_table_id"])

                if model_name != "bm25" and gold_table_id not in doc_id_set:
                    continue

                num_processed += 1

                q = _encode_query(model, model_name, question)

                K = 100
                top_idx, top_scores = batched_topk(
                    doc_embeddings, q, k=K, batch_size=8192
                )
                ranked_doc_ids = [doc_ids[i] for i in top_idx]
                ranked_scores  = [float(s) for s in top_scores]

                qid = str(question_id)
                qrels_dict[qid] = {gold_table_id: 1}
                runs_dict[qid]  = {
                    str(did): score
                    for did, score in zip(ranked_doc_ids, ranked_scores)
                }

            if num_processed == 0:
                print(f"Skipped:{perturbation} (no questions)", flush=True)
                if model_name != "bm25":
                    doc_embeddings = None
                    doc_ids        = None
                    doc_id_set     = None
                    gc.collect()
                continue

            metrics = evaluate(
                Qrels(qrels_dict), Run(runs_dict),
                RANX_METRICS, make_comparable=True,
            )

            all_results[str(perturbation)] = {
                **{m: metrics[m] for m in RANX_METRICS},
                "num_questions": num_processed,
                "num_tables":    len(doc_ids),
            }

            print(f"\t\tCompleted: {perturbation}", flush=True)
            print(
                f"\t\tQuestions: {num_processed}, Tables: {len(doc_ids)}",
                flush=True,
            )

            if model_name != "bm25":
                doc_embeddings = None
                doc_ids        = None
                doc_id_set     = None
                gc.collect()

        import json
        save_dir  = f"./data/retrieval_all/{model_name}_results/{dataset}/"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "perturbation_results.json")
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved results to: {save_path}\n")

        table_corpus = None
        gc.collect()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Usage:
    #   python -m retrieval.retrieval_all_multiple_dataset <model_name> [chunked]
    #
    # Add "chunked" as second arg to run the chunked pipeline (Steps 1-10).
    # Omit it to run the legacy flat retrieval.
    #
    # Set TOKENIZER_MODEL_NAMES to the HuggingFace IDs of the embedding models
    # you use so R(T) is conservative across all of them.
    #
    # Examples:
    #   CUDA_VISIBLE_DEVICES=5,6 python -m retrieval.retrieval_all_multiple_dataset reasonir chunked
    #   CUDA_VISIBLE_DEVICES=2   python -m retrieval.retrieval_all_multiple_dataset bge
    #   CUDA_VISIBLE_DEVICES=3   python -m retrieval.retrieval_all_multiple_dataset mpnet
    #   CUDA_VISIBLE_DEVICES=4   python -m retrieval.retrieval_all_multiple_dataset splade

    model_name   = sys.argv[1]
    chunked_mode = len(sys.argv) > 2 and sys.argv[2] == "chunked"

    accelerator = Accelerator()
    model = get_model(accelerator, model_name=model_name)

    all_questions_dict = get_data(
        "/data/Kushal/UniversalRetrieval_data/dataset_all/new_all_questions_dict_wtq_nqt_wikisql.pkl"
    )
    all_table_corpus = get_table_corpus(
        "/data/Kushal/UniversalRetrieval_data/dataset_all/all_table_transform_table_corpus.pkl"
    )

    if chunked_mode:
        retrieval_chunked(
            model_name=model_name,
            model=model,
            all_questions_dict=all_questions_dict,
            all_table_corpus=all_table_corpus,
            accelerator=accelerator,
        )
    else:
        retrieval(model_name, model, all_questions_dict, all_table_corpus, accelerator)


#CUDA_VISIBLE_DEVICES=5,6 python -m tbl_retrieval.retrieval.retrieval_all_multiple_dataset reasonir > result_reasonir.txt 2>&1
#CUDA_VISIBLE_DEVICES=2   python -m tbl_retrieval.retrieval.retrieval_all_multiple_dataset bge    > result_bge.txt 2>&1
#CUDA_VISIBLE_DEVICES=3   python -m tbl_retrieval.retrieval.retrieval_all_multiple_dataset mpnet  > result_mpnet.txt 2>&1
#CUDA_VISIBLE_DEVICES=4   python -m tbl_retrieval.retrieval.retrieval_all_multiple_dataset splade > result_splade.txt 2>&1