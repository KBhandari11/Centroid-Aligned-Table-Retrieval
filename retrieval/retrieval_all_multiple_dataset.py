import os
import sys
import json 
import pickle
from pandas import DataFrame
import pandas as pd
import pandera.pandas as pa
import pandas.io.json
from accelerate import Accelerator
import torch

import numpy as np
from ranx import evaluate, Qrels, Run

import re
import heapq
from ..utils.model import get_model
from ..utils.core import ( get_data,
                            get_table_corpus, 
                            compute_and_cache_rep_embeddings,
                            compute_and_cache_centroid,
                            batched_topk,
                            CACHE_ROOT
                        )
from ..utils.cache_embedding import *


def bm25_tokenize(text):
    if not isinstance(text, str):
        text = str(text)
    return re.findall(r"[A-Za-z0-9_]+|[|,]", text.lower())



def retrieval(model_name, model, all_questions_dict, all_table_corpus, accelerator):
    dataset_list = ["WTQ","WIKISQL","NQ"]
    dataset_list = ["WTQ","WIKISQL"]

    #dataset_list = ["NQ"]
    base_reps = [
        'pipe_serialized', 'token_serialized', 'space_serialized',
        "csv","tsv","html","markdown","latex","dict","json","xml",
        "shuffled_rows","shuffled_cols","transpose","mschema","macschema","ddl"
    ]
    centroid_reps = ["centroid_popular", "centroid_data", "centroid_structural", "centroid_schema", "centroid_all"]
    second_level = base_reps + centroid_reps

    for dataset in dataset_list:
        all_results = {}
        table_corpus = all_table_corpus[dataset]
        questions_dict = all_questions_dict[dataset]
        all_table_ids_list = list(table_corpus.keys())

        print(dataset, flush=True, file=sys.stderr)

        # Only main process builds caches + evaluates
        if not accelerator.is_main_process:
            accelerator.wait_for_everyone()
            continue

        # 1) Ensure base reps caches exist (compute once)
        if model_name != "bm25":
            for rep in base_reps:
                if not cache_exists(CACHE_ROOT, model_name, dataset, rep):
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

            # 2) Ensure centroid caches exist (streaming from base caches)
            # map  category dict -> centroid key
            category_map = {
                "centroid_popular": ["pipe_serialized", "token_serialized", "space_serialized"],
                "centroid_data": ["csv","tsv","html","markdown","latex","dict","json","xml"],
                "centroid_structural": ["shuffled_rows","shuffled_cols","transpose"],
                "centroid_schema": ["mschema","macschema","ddl"],
            }
            for cent_name, reps in category_map.items():
                if not cache_exists(CACHE_ROOT, model_name, dataset, cent_name):
                    print(f"[cache miss] building {cent_name}", flush=True, file=sys.stderr)
                    compute_and_cache_centroid(CACHE_ROOT, model_name, dataset, cent_name, reps)
                    gc.collect()

            # centroid_all across ALL base reps
            if not cache_exists(CACHE_ROOT, model_name, dataset, "centroid_all"):
                print("[cache miss] building centroid_all", flush=True, file=sys.stderr)
                compute_and_cache_centroid(CACHE_ROOT, model_name, dataset, "centroid_all", base_reps)
                gc.collect()

        accelerator.wait_for_everyone()

        # Now evaluate each perturbation by loading only that rep's embeddings
        for perturbation2 in second_level:
            print("\t perturbation", perturbation2, flush=True, file=sys.stderr)

            if model_name == "bm25":
                # unchanged bm25 path (but still keep docs list local and del after)
                docs, doc_ids = [], []
                for table_id in all_table_ids_list:
                    if table_id in table_corpus and perturbation2 in table_corpus[table_id]:
                        docs.append(table_corpus[table_id][perturbation2])
                        doc_ids.append(table_id)
                if len(doc_ids) == 0:
                    print(f"Skipped:{perturbation2} (no docs)", flush=True)
                    continue

            else:
                if not cache_exists(CACHE_ROOT, model_name, dataset, perturbation2):
                    print(f"Skipped:{perturbation2} (no cache)", flush=True)
                    continue

                doc_ids, doc_embeddings = load_cache_mmap(CACHE_ROOT, model_name, dataset, perturbation2)
                if len(doc_ids) == 0:
                    print(f"Skipped:{perturbation2} (no docs)", flush=True)
                    continue

                # Build a fast membership structure for gold lookup
                doc_id_set = set(doc_ids)

            qrels_dict = {}
            runs_dict = {}
            num_questions_processed = 0

            for question_id, question_data in questions_dict.items():
                question = str(question_data["question"])
                gold_table_id = str(question_data["gold_table_id"])

                if model_name != "bm25" and gold_table_id not in doc_id_set:
                    continue

                num_questions_processed += 1

                if model_name == "bm25":
                    pass

                # Encode query
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

                q = q.squeeze(0).astype(np.float32, copy=False)  # (D,)

                # Similarity: (N,D) @ (D,) -> (N,)
                # doc_embeddings is memmap; this avoids loading full matrix into RAM
                # Full ranking
                '''arr_scores = np.asarray(doc_embeddings @ q, dtype=np.float32)

                ranked_indices = np.argsort(arr_scores)[::-1]
                ranked_doc_ids = [doc_ids[i] for i in ranked_indices]
                ranked_scores = [float(arr_scores[i]) for i in ranked_indices]'''
                K = 100
                top_idx, top_scores = batched_topk(doc_embeddings, q, k=K, batch_size=8192)
                ranked_doc_ids = [doc_ids[i] for i in top_idx]
                ranked_scores = [float(s) for s in top_scores]

                qid = str(question_id)
                qrels_dict[qid] = {gold_table_id: 1}
                runs_dict[qid] = {str(did): score for did, score in zip(ranked_doc_ids, ranked_scores)}

            if num_questions_processed == 0:
                print(f"Skipped:{perturbation2} (no questions)", flush=True)
                # release this rep from RAM
                if model_name != "bm25":
                    doc_embeddings = None
                    doc_ids = None
                    doc_id_set = None
                    gc.collect()
                continue

            qrels = Qrels(qrels_dict)
            runs = Run(runs_dict)
            metrics = evaluate(
                qrels, runs,
                ["recall@1","recall@5","recall@10","recall@50","recall@100","recall@500",
                 "ndcg@1","ndcg@5","ndcg@10","ndcg@50"],
                make_comparable=True
            )

            all_results[str(perturbation2)] = {
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
                "num_tables": len(doc_ids) if model_name != "bm25" else len(doc_ids),
            }

            print(f"\t\tCompleted: {perturbation2}", flush=True)
            print(f"\t\tQuestions: {num_questions_processed}, Tables: {len(doc_ids)}", flush=True)

            # IMPORTANT: release per-rep memory/mmap refs
            if model_name != "bm25":
                doc_embeddings = None
                doc_ids = None
                doc_id_set = None
                gc.collect()

        # Save results
        import json, os
        save_dir = f"./data/retrieval_all/{model_name}_results/{dataset}/"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "perturbation_results.json")
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved results to: {save_path}\n")

        # release dataset-level refs
        table_corpus = None
        gc.collect()

if __name__ == "__main__":

    model_name = sys.argv[1] #"reasonir" or "splade"

    accelerator = Accelerator()
    # get ReasonIR model
    model = get_model(accelerator,model_name=model_name)

    all_questions_dict = get_data('./data/dataset_all/new_all_questions_dict_wtq_nqt_wikisql.pkl')
    table_corpus = get_table_corpus("./data/dataset_all/all_table_transform_table_corpus.pkl")

    retrieval(model_name, model, all_questions_dict,table_corpus,accelerator)
    #  CUDA_VISIBLE_DEVICES=5,6 python retrieval_all_multiple_dataset.py reasonir > result_reasonir.txt 2>&1
    #  CUDA_VISIBLE_DEVICES=2 python retrieval_all_multiple_dataset.py bge > result_bge.txt 2>&1
    #  CUDA_VISIBLE_DEVICES=3 python retrieval_all_multiple_dataset.py mpnet > result_mpnet.txt 2>&1
    #  CUDA_VISIBLE_DEVICES=4 python retrieval_all_multiple_dataset.py splade > result_splade.txt 2>&1