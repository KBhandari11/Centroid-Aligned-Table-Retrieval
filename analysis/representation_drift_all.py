#!/usr/bin/env python3
"""
Compute representation drift (magnitude drift) for table serializations.

Uses:
  - Model loading (Accelerate):
        accelerator = Accelerator()
        model = get_model(accelerator, model_name=...)
  - Table corpus loading (pickle):
        table_corpus = get_table_corpus("./all_table_transform_table_corpus.pkl")

Expected table_corpus structure:
  table_corpus[DATASET][TABLE_ID][REPRESENTATION] -> str (or convertible to str)

Outputs:
  - drift_<model_name>.csv                 (per table_id x representation)
  - drift_summary_<model_name>.csv         (mean drift per dataset x representation)
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from accelerate import Accelerator

# Your project utilities (must provide: get_model, encode_texts_in_batches)
from retrieval_all_multiple_dataset import *

category = {
    "Popular Representation": ["pipe_serialized", "token_serialized", "space_serialized"],
    "Data Representation": ["csv", "tsv", "html", "markdown", "latex", "dict", "json", "xml"],
    "Structural Transformations": ["shuffled_rows", "shuffled_cols", "transpose"],
    "Schema and Definition Types": ["mschema", "macschema", "ddl"],
}
# This is your "centroid_all" store key (FINAL centroid across categories)
FINAL_CENTROID_KEY = "centroid_all"


# ----------------------------
# Loading helpers (as you provided)
# ----------------------------
def get_table_corpus(save_table_real_data_file_path: str):
    all_tables = {}
    if os.path.exists(save_table_real_data_file_path):
        with open(save_table_real_data_file_path, "rb") as f:
            all_tables = pickle.load(f)
    return all_tables


# ----------------------------
# Math helpers
# ----------------------------
def _to_text(x):
    return x if isinstance(x, str) else str(x)

def _l2_normalize_rows_torch(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return X / (X.norm(dim=1, keepdim=True) + eps)

def _build_rep_to_category(category: dict[str, list[str]]) -> dict[str, str]:
    rep_to_cat = {}
    for cat, reps in category.items():
        for r in reps:
            rep_to_cat[r] = cat
    return rep_to_cat

def build_rep_to_centroid_key(category_map: dict, centroid_keys: dict) -> dict[str, str]:
    """
    Returns rep -> centroid_key (e.g., "csv" -> "centroid_data")
    """
    rep_to_centroid = {}
    for cat_name, rep_list in category_map.items():
        ck = centroid_keys[cat_name]
        for r in rep_list:
            rep_to_centroid[r] = ck
    return rep_to_centroid

def build_id_to_row(store_entry: dict) -> dict[str, int]:
    """
    store_entry: {"doc_ids": [...], "emb": Tensor(n, D)}
    """
    return {tid: i for i, tid in enumerate(store_entry["doc_ids"])}


# ----------------------------
# Drift computation
# ----------------------------
@torch.no_grad()
@torch.no_grad()
def compute_drift_from_stores(
    rep_store: dict,
    centroid_store: dict,
    category_map: dict,
    centroid_keys: dict,
    final_centroid_key: str,
    normalize_cosine: bool = True,
    l2_on_unit_sphere: bool = False,
) -> pd.DataFrame:
    """
    For each table_id x representation:
      - drift to its category centroid
      - drift to centroid_all (final centroid key)

    Returns columns:
      table_id, representation, category_name, category_centroid_key,
      drift_cos_to_cat, drift_l2_to_cat, drift_cos_to_all, drift_l2_to_all
    """
    rep_to_cat = _build_rep_to_category(category_map)
    rep_to_centroid = build_rep_to_centroid_key(category_map, centroid_keys)

    if final_centroid_key not in centroid_store:
        raise ValueError(
            f"final_centroid_key={final_centroid_key} not in centroid_store keys={list(centroid_store.keys())}"
        )

    all_entry = centroid_store[final_centroid_key]
    all_id2row = build_id_to_row(all_entry)
    all_emb = all_entry["emb"]  # (n_all, D)

    # id->row for each centroid key
    centroid_id2row = {}
    for cat_name, ck in centroid_keys.items():
        if ck in centroid_store:
            centroid_id2row[ck] = build_id_to_row(centroid_store[ck])

    rows = []

    for rep, rep_entry in rep_store.items():
        if rep not in rep_to_centroid:
            continue

        cat_name = rep_to_cat.get(rep, None)
        ck = rep_to_centroid[rep]

        if ck not in centroid_store:
            continue

        cat_entry = centroid_store[ck]
        cat_id2row = centroid_id2row[ck]
        cat_emb = cat_entry["emb"]

        rep_ids = rep_entry["doc_ids"]
        rep_emb = rep_entry["emb"]  # (n_rep, D)

        # intersection across: rep, cat centroid, final centroid
        common_tids = []
        rep_rows = []
        cat_rows = []
        all_rows = []

        for i, tid in enumerate(rep_ids):
            j = cat_id2row.get(tid, None)
            k = all_id2row.get(tid, None)
            if j is None or k is None:
                continue
            common_tids.append(tid)
            rep_rows.append(i)
            cat_rows.append(j)
            all_rows.append(k)

        if len(common_tids) == 0:
            continue

        R = rep_emb[rep_rows]
        Ccat = cat_emb[cat_rows]
        Call = all_emb[all_rows]

        # cosine drift
        if normalize_cosine:
            Rn = _l2_normalize_rows_torch(R)
            Ccatn = _l2_normalize_rows_torch(Ccat)
            Calln = _l2_normalize_rows_torch(Call)
        else:
            Rn, Ccatn, Calln = R, Ccat, Call

        cos_sim_cat = (Rn * Ccatn).sum(dim=1).clamp(-1, 1)
        cos_sim_all = (Rn * Calln).sum(dim=1).clamp(-1, 1)

        drift_cos_to_cat = (1.0 - cos_sim_cat).cpu().numpy()
        drift_cos_to_all = (1.0 - cos_sim_all).cpu().numpy()

        # L2 drift
        if l2_on_unit_sphere:
            Rl = _l2_normalize_rows_torch(R)
            Ccatl = _l2_normalize_rows_torch(Ccat)
            Calll = _l2_normalize_rows_torch(Call)
        else:
            Rl, Ccatl, Calll = R, Ccat, Call

        drift_l2_to_cat = (Rl - Ccatl).norm(dim=1).cpu().numpy()
        drift_l2_to_all = (Rl - Calll).norm(dim=1).cpu().numpy()

        for tid, dc, dl, dca, dla in zip(common_tids, drift_cos_to_cat, drift_l2_to_cat, drift_cos_to_all, drift_l2_to_all):
            rows.append(
                {
                    "table_id": tid,
                    "representation": rep,
                    "category_name": cat_name,
                    "category_centroid_key": ck,
                    "drift_cos_to_cat": float(dc),
                    "drift_l2_to_cat": float(dl),
                    "drift_cos_to_all": float(dca),
                    "drift_l2_to_all": float(dla),
                    "normalize_cosine": normalize_cosine,
                    "l2_on_unit_sphere": l2_on_unit_sphere,
                }
            )

    return pd.DataFrame(rows)

@torch.no_grad()
def compute_centroid_drift_to_all(
    centroid_store: dict,
    centroid_keys: dict,          # category_name -> centroid_key
    final_centroid_key: str,      # centroid_all key
    normalize_cosine: bool = True,
    l2_on_unit_sphere: bool = False,
) -> pd.DataFrame:
    """
    For each table_id where both exist:
      drift(centroid_<category>, centroid_all)

    Returns:
      table_id, category_name, centroid_key, drift_cos_cat_to_all, drift_l2_cat_to_all
    """
    if final_centroid_key not in centroid_store:
        raise ValueError(
            f"final_centroid_key={final_centroid_key} not in centroid_store keys={list(centroid_store.keys())}"
        )

    all_entry = centroid_store[final_centroid_key]
    all_id2row = build_id_to_row(all_entry)
    all_emb = all_entry["emb"]

    rows = []

    for cat_name, ck in centroid_keys.items():
        if ck not in centroid_store:
            continue

        cat_entry = centroid_store[ck]
        cat_id2row = build_id_to_row(cat_entry)
        cat_emb = cat_entry["emb"]

        common_tids, cat_rows, all_rows = [], [], []
        for tid in cat_entry["doc_ids"]:
            j = cat_id2row.get(tid, None)
            k = all_id2row.get(tid, None)
            if j is None or k is None:
                continue
            common_tids.append(tid)
            cat_rows.append(j)
            all_rows.append(k)

        if len(common_tids) == 0:
            continue

        Ccat = cat_emb[cat_rows]
        Call = all_emb[all_rows]

        # cosine drift
        if normalize_cosine:
            Ccatn = _l2_normalize_rows_torch(Ccat)
            Calln = _l2_normalize_rows_torch(Call)
        else:
            Ccatn, Calln = Ccat, Call

        cos_sim = (Ccatn * Calln).sum(dim=1).clamp(-1, 1)
        drift_cos = (1.0 - cos_sim).cpu().numpy()

        # L2 drift
        if l2_on_unit_sphere:
            Ccatl = _l2_normalize_rows_torch(Ccat)
            Calll = _l2_normalize_rows_torch(Call)
        else:
            Ccatl, Calll = Ccat, Call

        drift_l2 = (Ccatl - Calll).norm(dim=1).cpu().numpy()

        for tid, dc, dl in zip(common_tids, drift_cos, drift_l2):
            rows.append(
                {
                    "table_id": tid,
                    "category_name": cat_name,
                    "centroid_key": ck,
                    "drift_cos_cat_to_all": float(dc),
                    "drift_l2_cat_to_all": float(dl),
                    "normalize_cosine": normalize_cosine,
                    "l2_on_unit_sphere": l2_on_unit_sphere,
                }
            )

    return pd.DataFrame(rows)



def summarize_drift_by_category(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby(["dataset", "category_name", "representation", "category_centroid_key"], as_index=False)
          .agg(
              mean_cos_cat=("drift_cos_to_cat", "mean"),
              std_cos_cat=("drift_cos_to_cat", "std"),
              mean_l2_cat=("drift_l2_to_cat", "mean"),
              std_l2_cat=("drift_l2_to_cat", "std"),
              mean_cos_all=("drift_cos_to_all", "mean"),
              std_cos_all=("drift_cos_to_all", "std"),
              mean_l2_all=("drift_l2_to_all", "mean"),
              std_l2_all=("drift_l2_to_all", "std"),
              n=("table_id", "count"),
          )
          .sort_values(["dataset", "mean_cos_all"], ascending=[True, False])
    )

def summarize_centroid_drift(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby(["dataset", "category_name", "centroid_key"], as_index=False)
          .agg(
              mean_cos_cat_to_all=("drift_cos_cat_to_all", "mean"),
              std_cos_cat_to_all=("drift_cos_cat_to_all", "std"),
              mean_l2_cat_to_all=("drift_l2_cat_to_all", "mean"),
              std_l2_cat_to_all=("drift_l2_cat_to_all", "std"),
              n=("table_id", "count"),
          )
          .sort_values(["dataset", "mean_cos_cat_to_all"], ascending=[True, False])
    )


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    table_corpus_path = "/data/Kushal/UniversalRetrieval_data/dataset_all/all_table_transform_table_corpus.pkl"
    table_corpus = get_table_corpus(table_corpus_path)
    if not table_corpus:
        raise RuntimeError(f"Loaded empty table_corpus from {table_corpus_path}")

    model_name = sys.argv[1]
    accelerator = Accelerator()
    model = get_model(accelerator, model_name=model_name)

    representations = [
        "pipe_serialized", "token_serialized", "space_serialized",
        "csv", "tsv", "html", "markdown", "latex", "dict", "json", "xml",
        "shuffled_rows", "shuffled_cols", "transpose",
        "mschema", "macschema", "ddl",
    ]

    datasets = ["WTQ", "WIKISQL"]

    all_rep_df = []
    all_cent_df = []

    for ds in datasets:
        if ds not in table_corpus:
            print(f"[warn] dataset {ds} not in table_corpus, skipping")
            continue

        table_ids = list(table_corpus[ds].keys())

        rep_store, centroid_store = precompute_rep_embeddings_and_centroids(
            model_name=model_name,
            model=model,
            table_corpus=table_corpus[ds],
            table_ids=table_ids,
            category_map=category,
            batch_size=256,
            instruction="",
            normalize=False,
        )

        # reps -> (cat centroid, centroid_all)
        df_ds = compute_drift_from_stores(
            rep_store=rep_store,
            centroid_store=centroid_store,
            category_map=category,
            centroid_keys=CENTROID_KEYS,
            final_centroid_key=FINAL_CENTROID_KEY,
            normalize_cosine=True,
            l2_on_unit_sphere=False,
        )
        if df_ds is not None and not df_ds.empty:
            df_ds.insert(0, "dataset", ds)
            all_rep_df.append(df_ds)

        # centroid_<category> -> centroid_all
        df_cent = compute_centroid_drift_to_all(
            centroid_store=centroid_store,
            centroid_keys=CENTROID_KEYS,
            final_centroid_key=FINAL_CENTROID_KEY,
            normalize_cosine=True,
            l2_on_unit_sphere=False,
        )
        if df_cent is not None and not df_cent.empty:
            df_cent.insert(0, "dataset", ds)
            all_cent_df.append(df_cent)

    if not all_rep_df and not all_cent_df:
        raise RuntimeError("No drift rows produced. Check corpus / centroids / embedding.")

    out_dir = f"/data/Kushal/UniversalRetrieval_data/retrieval_all/{model_name}_results"
    os.makedirs(out_dir, exist_ok=True)

    # ---- Save rep drift
    if all_rep_df:
        df_rep = pd.concat(all_rep_df, ignore_index=True)
        out_csv = os.path.join(out_dir, f"drift_by_category.csv")
        df_rep.to_csv(out_csv, index=False)

        summ_rep = summarize_drift_by_category(df_rep)
        out_summ = os.path.join(out_dir, f"drift_by_category_summary.csv")
        summ_rep.to_csv(out_summ, index=False)

        print(f"[done] wrote: {out_csv}")
        print(f"[done] wrote: {out_summ}")
        print(summ_rep.head(20))

    # ---- Save centroid drift
    if all_cent_df:
        df_cent_all = pd.concat(all_cent_df, ignore_index=True)
        out_csv2 = os.path.join(out_dir, f"centroid_drift_to_all.csv")
        df_cent_all.to_csv(out_csv2, index=False)

        summ_cent = summarize_centroid_drift(df_cent_all)
        out_summ2 = os.path.join(out_dir, f"centroid_drift_to_all_summary.csv")
        summ_cent.to_csv(out_summ2, index=False)

        print(f"[done] wrote: {out_csv2}")
        print(f"[done] wrote: {out_summ2}")
        print(summ_cent.head(20))
    #CUDA_VISIBLE_DEVICES=3,7 python representation_drift.py splade > result_repr.txt 2>&1

