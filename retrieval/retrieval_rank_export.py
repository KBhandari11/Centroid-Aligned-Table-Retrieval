import os
import sys
import gc
import json
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator

from ..utils.model import get_model
from ..utils.core import ( get_data,
                            get_table_corpus, 
                            compute_and_cache_rep_embeddings,
                            compute_and_cache_centroid,batched_dot_scores,
                            CACHE_ROOT
                        )
from ..utils.cache_embedding import *

base_reps = [
    "pipe_serialized", "token_serialized", "space_serialized",
    "csv", "tsv", "html", "markdown", "latex", "dict", "json", "xml",
    "shuffled_rows", "shuffled_cols", "transpose",
    "mschema", "macschema", "ddl",
]
centroid_reps = [
    "centroid_popular", "centroid_data", "centroid_structural", "centroid_schema", "centroid_all"
]
all_reps = base_reps + centroid_reps

category_map = {
    "centroid_popular": ["pipe_serialized", "token_serialized", "space_serialized"],
    "centroid_data": ["csv", "tsv", "html", "markdown", "latex", "dict", "json", "xml"],
    "centroid_structural": ["shuffled_rows", "shuffled_cols", "transpose"],
    "centroid_schema": ["mschema", "macschema", "ddl"],
}


def get_rank(model,model_name, dataset_list, all_questions_dict, all_table_corpus):
  for dataset in dataset_list:
      print(f"\n[Dataset] {dataset}", flush=True)

      table_corpus = all_table_corpus[dataset]
      questions_dict = all_questions_dict[dataset]
      all_table_ids_list = list(table_corpus.keys())

      # Ensure caches for non-BM25 models
      if model_name != "bm25":
          for rep in base_reps:
              if not cache_exists(CACHE_ROOT, model_name, dataset, rep):
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
              if not cache_exists(CACHE_ROOT, model_name, dataset, cent_name):
                  print(f"[cache miss] building {dataset}/{cent_name}", flush=True)
                  compute_and_cache_centroid(CACHE_ROOT, model_name, dataset, cent_name, reps)
                  gc.collect()

          if not cache_exists(CACHE_ROOT, model_name, dataset, "centroid_all"):
              print(f"[cache miss] building {dataset}/centroid_all", flush=True)
              compute_and_cache_centroid(CACHE_ROOT, model_name, dataset, "centroid_all", base_reps)
              gc.collect()

      for perturbation in all_reps:
          print(f"  [Rep] {perturbation}", flush=True)
          
          if not cache_exists(CACHE_ROOT, model_name, dataset, perturbation):
              print(f"    skipped (no cache): {perturbation}", flush=True)
              continue

          doc_ids, doc_embeddings = load_cache_mmap(CACHE_ROOT, model_name, dataset, perturbation)
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

              # Query embedding (same logic as your retrieval script)
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

              q = q.squeeze(0).astype(np.float32, copy=False)

              # full scores needed for exact gold rank
              scores = batched_dot_scores(doc_embeddings, q, batch_size=8192)
              gold_idx = doc_index[gold_table_id]
              gold_score = float(scores[gold_idx])

              # exact rank
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

          # cleanup memmap refs
          doc_embeddings = None
          doc_ids = None
          gc.collect()

          # save per-perturbation CSV
          dataset_dir = os.path.join(output_root, dataset)
          os.makedirs(dataset_dir, exist_ok=True)

          rep_csv = os.path.join(dataset_dir, f"{perturbation}_gold_rank_per_question.csv")
          rep_df.to_csv(rep_csv, index=False)
          print(f"    saved: {rep_csv}", flush=True)


  print("\nDone.", flush=True)



if __name__ == "__main__":
    
    #-----------------------------
    # Config
    # -----------------------------
    model_name = sys.argv[1]  # reasonir / rank1 / bge / mpnet / jina / splade / bm25
    accelerator = Accelerator()

    dataset_list = ["WTQ", "WIKISQL","NQ"]  # matches your current retrieval() setup

    questions_path = "./data/dataset_all/new_all_questions_dict_wtq_nqt_wikisql.pkl"
    tables_path = "./data/dataset_all/all_table_transform_table_corpus.pkl"

    output_root = f"./data/retrieval_all/{model_name}_results_rank"
    os.makedirs(output_root, exist_ok=True)


    # -----------------------------
    # Load model + data
    # -----------------------------
    model = get_model(accelerator, model_name=model_name)

    all_questions_dict = get_data(questions_path)
    all_table_corpus = get_table_corpus(tables_path)

    # Main-process only: same pattern as your script
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        sys.exit(0)

    get_rank(model,model_name, dataset_list, all_questions_dict, all_table_corpus)

    #  CUDA_VISIBLE_DEVICES=5,6 python retrieval_rank_export.py reasonir > result_reasonir.txt 2>&1
    #  CUDA_VISIBLE_DEVICES=2 python retrieval_rank_export.py bge > result_bge.txt 2>&1
    #  CUDA_VISIBLE_DEVICES=3 python retrieval_rank_export.py mpnet > result_mpnet.txt 2>&1
    #  CUDA_VISIBLE_DEVICES=4 python retrieval_rank_export.py splade > result_splade.txt 2>&1
    