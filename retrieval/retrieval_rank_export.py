import os
import sys
import gc
import json
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator

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
    batched_dot_scores,
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
all_reps = base_reps + centroid_reps

category_map = {
    "centroid_popular":    ["pipe_serialized", "token_serialized", "space_serialized"],
    "centroid_data":       ["csv", "tsv", "html", "markdown", "latex", "dict", "json", "xml"],
    "centroid_structural": ["shuffled_rows", "shuffled_cols", "transpose"],
    "centroid_schema":     ["mschema", "macschema", "ddl"],
}

# Step 10: NQ-Tables uses only this subset (selected greedily for complementarity)
NQ_FORMATS = ["csv", "markdown", "token_serialized", "ddl"]


# ── Chunked retrieval (Steps 7-8) ─────────────────────────────────────────────

def get_rank_chunked(model, model_name, dataset_list, all_questions_dict,
                     all_table_corpus, output_root, run_sanity_check=True):
    """
    Steps 7-10: encode chunks, score with top-3 mean, export per-question ranks.

    The tokenizer is extracted directly from the loaded model via
    get_tokenizer_from_model — no separate model name list needed.
    """
    tokenizers = get_tokenizer_from_model(model, model_name)

    for dataset in dataset_list:
        print(f"\n[Dataset] {dataset}", flush=True)

        table_corpus = all_table_corpus[dataset]
        questions_dict = all_questions_dict[dataset]

        # ── Step 4: build or load manifest ───────────────────────────────────
        manifest_path = os.path.join(
            output_root, dataset, f"{dataset}_manifest.json"
        )
        if os.path.exists(manifest_path):
            print(f"  Loading manifest: {manifest_path}", flush=True)
            manifest = load_manifest(manifest_path)
        else:
            print(f"  Building manifest for {dataset} …", flush=True)
            manifest = build_manifest(
                table_corpus=table_corpus,
                dataset=dataset,
                tokenizers=tokenizers,
                manifest_path=manifest_path,
            )

        # ── Sanity check (run before any encoding) ────────────────────────────
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
                print(f"  [cache miss] building chunked {fmt}", flush=True)
                compute_and_cache_chunked_embeddings(
                    model_name=model_name,
                    model=model,
                    table_corpus=table_corpus,
                    manifest=manifest,
                    format_name=fmt,
                    cache_root=CACHE_ROOT,
                    dataset=dataset,
                    batch_size=256 if "qwen" not in model_name else 8,
                    instruction="",
                    normalize=False,
                )
                gc.collect()

        # ── Chunked centroids (average chunked embeddings within each category) ─
        chunked_category_map = {
            f"chunked_centroid_{k}": [f"chunked_{f}" for f in v]
            for k, v in category_map.items()
        }
        chunked_category_map["chunked_centroid_all"] = [f"chunked_{f}" for f in formats]

        for cent_name, rep_list in chunked_category_map.items():
            if not cache_exists_for_model(CACHE_ROOT, model_name, dataset, cent_name):
                # Only compute if all constituent caches exist
                if all(cache_exists_for_model(CACHE_ROOT, model_name, dataset, r) for r in rep_list):
                    print(f"  [cache miss] building {cent_name}", flush=True)
                    compute_and_cache_centroid(
                        CACHE_ROOT, model_name, dataset, cent_name, rep_list
                    )
                    gc.collect()
                else:
                    print(f"  [skip centroid] {cent_name}: missing constituent caches", flush=True)

        # ── Step 8: score each question ───────────────────────────────────────
        score_reps = [(f"chunked_{fmt}", fmt) for fmt in formats] + [
            (name, name) for name in chunked_category_map
        ]
        for rep_key, _ in score_reps:
            dataset_dir = os.path.join(output_root, dataset)
            rep_csv = os.path.join(dataset_dir, f"{rep_key}_gold_rank_per_question.csv")

            if os.path.exists(rep_csv):
                print(f"  [Rep] {rep_key}: already scored, skipping", flush=True)
                continue

            print(f"  [Rep] {rep_key}", flush=True)

            if not cache_exists_for_model(CACHE_ROOT, model_name, dataset, rep_key):
                print(f"    skipped (no cache): {rep_key}", flush=True)
                continue

            doc_ids, doc_embeddings = load_cache_for_model(
                CACHE_ROOT, model_name, dataset, rep_key
            )
            if len(doc_ids) == 0:
                print(f"    skipped (empty): {rep_key}", flush=True)
                continue

            rep_rows = []
            for qid, qdata in questions_dict.items():
                question = str(qdata["question"])
                gold_table_id = str(qdata["gold_table_id"])

                # Encode query (unchunked, unmodified — Step 7)
                if model_name == "reasonir":
                    q = model.encode([question], instruction="")
                elif model_name == "jina":
                    q = model.encode(
                        [question],
                        task="retrieval.query",
                        prompt_name="retrieval.query",
                    )
                elif model_name == "splade":
                    q = model.encode_query([question]).to_dense()
                else:
                    q = model.encode([question])

                if isinstance(q, torch.Tensor):
                    q = q.detach().cpu().float().numpy()
                else:
                    q = np.asarray(q, dtype=np.float32)
                q = q.squeeze(0).astype(np.float32, copy=False)

                # Step 8: top-3 chunk mean → ranked table list
                ranked = score_top3_chunk_mean(
                    doc_ids, doc_embeddings, q, k=len(manifest["tables"])
                )
                # ranked = [(table_id, primary_score, max_score), ...]

                primary_scores   = {r[0]: r[1] for r in ranked}
                max_scores       = {r[0]: r[2] for r in ranked}

                if gold_table_id not in primary_scores:
                    continue

                gold_primary = primary_scores[gold_table_id]
                gold_max     = max_scores[gold_table_id]

                # Exact rank by primary score
                rank = int(
                    sum(1 for r in ranked if r[1] > gold_primary) + 1
                )

                rep_rows.append({
                    "model":         model_name,
                    "dataset":       dataset,
                    "format":        rep_key,
                    "question_id":   str(qid),
                    "gold_table_id": gold_table_id,
                    "rank":          rank,
                    "gold_score_top3_mean": gold_primary,
                    "gold_score_max":       gold_max,
                    "num_tables":    len(manifest["tables"]),
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
            os.makedirs(dataset_dir, exist_ok=True)
            rep_df.to_csv(rep_csv, index=False)
            print(f"    saved: {rep_csv}", flush=True)

    print("\nDone.", flush=True)


# ── Legacy flat retrieval (unchanged, kept for comparison) ────────────────────

def get_rank(model, model_name, dataset_list, all_questions_dict, all_table_corpus):
    for dataset in dataset_list:
        print(f"\n[Dataset] {dataset}", flush=True)

        table_corpus = all_table_corpus[dataset]
        questions_dict = all_questions_dict[dataset]
        all_table_ids_list = list(table_corpus.keys())

        if model_name != "bm25":
            for rep in base_reps:
                if not cache_exists_for_model(CACHE_ROOT, model_name, dataset, rep):
                    print(f"[cache miss] building {dataset}/{rep}", flush=True)
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
                    print(f"[cache miss] building {dataset}/{cent_name}", flush=True)
                    compute_and_cache_centroid(
                        CACHE_ROOT, model_name, dataset, cent_name, reps
                    )
                    gc.collect()

            if not cache_exists_for_model(CACHE_ROOT, model_name, dataset, "centroid_all"):
                print(f"[cache miss] building {dataset}/centroid_all", flush=True)
                compute_and_cache_centroid(
                    CACHE_ROOT, model_name, dataset, "centroid_all", base_reps
                )
                gc.collect()

        for perturbation in all_reps:
            print(f"  [Rep] {perturbation}", flush=True)

            if not cache_exists_for_model(CACHE_ROOT, model_name, dataset, perturbation):
                print(f"    skipped (no cache): {perturbation}", flush=True)
                continue

            doc_ids, doc_embeddings = load_cache_for_model(
                CACHE_ROOT, model_name, dataset, perturbation
            )
            if len(doc_ids) == 0:
                print(f"    skipped (empty doc list): {perturbation}", flush=True)
                continue

            doc_index = {str(d): i for i, d in enumerate(doc_ids)}

            rep_rows = []
            for qid, qdata in questions_dict.items():
                question = str(qdata["question"])
                gold_table_id = str(qdata["gold_table_id"])

                if gold_table_id not in doc_index:
                    continue

                if model_name == "reasonir":
                    q = model.encode([question], instruction="")
                elif model_name == "jina":
                    q = model.encode(
                        [question],
                        task="retrieval.query",
                        prompt_name="retrieval.query",
                    )
                elif model_name == "splade":
                    q = model.encode_query([question]).to_dense()
                else:
                    q = model.encode([question])

                if isinstance(q, torch.Tensor):
                    q = q.detach().cpu().float().numpy()
                else:
                    q = np.asarray(q, dtype=np.float32)
                q = q.squeeze(0).astype(np.float32, copy=False)

                scores = batched_dot_scores(doc_embeddings, q, batch_size=8192)
                gold_idx = doc_index[gold_table_id]
                gold_score = float(scores[gold_idx])
                rank = int(np.sum(scores > gold_score) + 1)

                rep_rows.append({
                    "model":         model_name,
                    "dataset":       dataset,
                    "perturbation":  perturbation,
                    "question_id":   str(qid),
                    "gold_table_id": gold_table_id,
                    "rank":          rank,
                    "gold_score":    gold_score,
                    "num_docs":      len(doc_ids),
                    "hit@1":   int(rank <= 1),
                    "hit@5":   int(rank <= 5),
                    "hit@10":  int(rank <= 10),
                    "hit@50":  int(rank <= 50),
                    "hit@100": int(rank <= 100),
                })

            if not rep_rows:
                print(f"    skipped (no usable questions): {perturbation}", flush=True)
                doc_embeddings = None
                doc_ids = None
                gc.collect()
                continue

            rep_df = pd.DataFrame(rep_rows)
            doc_embeddings = None
            doc_ids = None
            gc.collect()

            dataset_dir = os.path.join(output_root, dataset)
            os.makedirs(dataset_dir, exist_ok=True)
            rep_csv = os.path.join(
                dataset_dir, f"{perturbation}_gold_rank_per_question.csv"
            )
            rep_df.to_csv(rep_csv, index=False)
            print(f"    saved: {rep_csv}", flush=True)

    print("\nDone.", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Usage:
    #   python retrieval_rank_export.py <model_name> [chunked]
    #
    # Add "chunked" as second arg to run the chunked pipeline (Steps 1-10).
    # Omit it to run the legacy flat retrieval.
    #
    # Tokenizer model names: edit TOKENIZER_MODEL_NAMES to match the embedding
    # models you intend to use.  The R(T) budget will be conservative across
    # all of them.
    #
    # Examples:
    #   CUDA_VISIBLE_DEVICES=5,6 python -m retrieval.retrieval_rank_export reasonir chunked
    #   CUDA_VISIBLE_DEVICES=2   python -m retrieval.retrieval_rank_export bge

    model_name = sys.argv[1]
    chunked_mode = len(sys.argv) > 2 and sys.argv[2] == "chunked"

    accelerator = Accelerator()

    dataset_list = ["WTQ", "WIKISQL", "NQ"]

    questions_path = "/data/Kushal/UniversalRetrieval_data/dataset_all/new_all_questions_dict_wtq_nqt_wikisql.pkl"
    tables_path    = "/data/Kushal/UniversalRetrieval_data/dataset_all/all_table_transform_table_corpus.pkl"
    output_root    = f"/data/Kushal/UniversalRetrieval_data/retrieval_all/{model_name}_results_rank"
    os.makedirs(output_root, exist_ok=True)

    model = get_model(accelerator, model_name=model_name)

    all_questions_dict = get_data(questions_path)
    all_table_corpus   = get_table_corpus(tables_path)

    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        sys.exit(0)

    if chunked_mode:
        get_rank_chunked(
            model=model,
            model_name=model_name,
            dataset_list=dataset_list,
            all_questions_dict=all_questions_dict,
            all_table_corpus=all_table_corpus,
            output_root=output_root,
        )
    else:
        get_rank(
            model=model,
            model_name=model_name,
            dataset_list=dataset_list,
            all_questions_dict=all_questions_dict,
            all_table_corpus=all_table_corpus,
        )
#CUDA_VISIBLE_DEVICES=5,6 python -m tbl_retrieval.retrieval.retrieval_rank_export reasonir chunked > result_reasonir_rank_chunked.txt 2>&1