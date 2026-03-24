"""
create_representations.py

Loads question and table data from disk, generates multiple serialized
representations of each table, and saves the result as a pickle.

Representations produced per table:
    csv, tsv, html, markdown, latex, dict, json, xml,
    shuffled_rows, shuffled_cols, mschema, macschema, ddl, transpose,
    pipe_serialized, token_serialized, space_serialized
"""

import pickle
import unicodedata
import warnings

import pandas as pd
import pandas.io.json
import pandera.pandas as pa
from pandas import DataFrame

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ============================================================
# Paths
# ============================================================

QUESTION_DATA_PATH  = "./data/dataset_all/all_questions_dict_full_wtq_nq_wikisql"
TABLES_DATA_PATH    = "./data/dataset_all/all_tables_full_wtq_nq_wikisql"
SAVE_QUESTIONS_PATH = "./data/dataset_all/new_all_questions_dict_wtq_nqt_wikisql.pkl"
SAVE_TABLES_PATH    = "./data/dataset_all/all_table_transform_table_corpus.pkl"

DATASETS = ["WTQ", "NQ", "WIKISQL"]


# ============================================================
# Column sanitization
# ============================================================

def sanitize_and_dedupe(cols, sep="#", empty_name="Column"):
    """
    Normalizes column names to NFKC Unicode, strips whitespace, prefixes
    digit-leading names, and appends a counter suffix to duplicates.
    """
    cleaned = []
    for c in cols:
        s = unicodedata.normalize("NFKC", "" if c is None else str(c)).strip()
        if not s:
            s = empty_name
        elif s[0].isdigit():
            s = f"{empty_name} {s}"
        cleaned.append(s)

    seen = {}
    out = []
    for s in cleaned:
        seen[s] = seen.get(s, 0) + 1
        out.append(s if seen[s] == 1 else f"{s} {sep}{seen[s]}")
    return out


# ============================================================
# Serializers
# ============================================================

def get_csv(df: DataFrame) -> str:
    return df.to_csv()


def get_tsv(df: DataFrame, **kwargs) -> str:
    kwargs.setdefault("sep", "\t")
    return df.to_csv(**kwargs)


def get_html(df: DataFrame) -> str:
    return df.to_html()


def get_markdown(df: DataFrame) -> str:
    return df.to_markdown()


def get_latex(df: DataFrame) -> str:
    return df.to_latex()


def get_dict(df: DataFrame) -> dict:
    return df.to_dict()


def get_json(df: DataFrame) -> str:
    return df.to_json()


def get_xml(df: DataFrame, parser: str = "etree") -> str:
    return df.to_xml(parser=parser, pretty_print=False)


def get_shuffled_rows(df: DataFrame) -> DataFrame:
    return df.sample(frac=1).reset_index(drop=True)


def get_shuffled_cols(df: DataFrame) -> DataFrame:
    return df.sample(frac=1, axis=1)


def get_mschema(df: DataFrame) -> dict:
    """
    Builds a pandera DataFrameSchema from inferred column dtypes and
    returns it alongside the row records.
    """
    columns = {}
    for col in df.columns:
        s = df[col]
        dtype = s.dtype
        if pd.api.types.is_string_dtype(dtype) or dtype == object:
            pa_dtype = pa.String
        elif pd.api.types.is_integer_dtype(dtype):
            pa_dtype = pa.Int
        elif pd.api.types.is_float_dtype(dtype):
            pa_dtype = pa.Float
        elif pd.api.types.is_bool_dtype(dtype):
            pa_dtype = pa.Bool
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            pa_dtype = pa.DateTime
        else:
            pa_dtype = pa.Object
        columns[col] = pa.Column(pa_dtype)

    return {
        "schema": pa.DataFrameSchema(columns),
        "data":   df.to_dict(orient="records"),
    }


def get_macschema(df: DataFrame) -> dict:
    """
    Uses pandas JSON Table Schema builder for metadata-aware schema,
    then attaches the row records.
    """
    table_schema = pd.io.json.build_table_schema(df)
    table_schema["data"] = df.to_dict(orient="records")
    return table_schema


def get_ddl(df: DataFrame, table_name: str = "Table") -> str:
    """
    Produces a CREATE TABLE statement followed by INSERT statements
    for every row. Bool values become 0/1; strings are single-quote escaped.
    """
    dtype_map = {
        "int64":          "INTEGER",
        "float64":        "FLOAT",
        "object":         "TEXT",
        "bool":           "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP",
    }

    ddl_str = f"CREATE TABLE {table_name} (\n"
    for col in df.columns:
        sql_type = dtype_map.get(str(df[col].dtype), "TEXT")
        ddl_str += f"  {col} {sql_type},\n"
    ddl_str = ddl_str.rstrip(",\n") + "\n);\n"

    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            val = row[col]
            if pd.isnull(val):
                values.append("NULL")
            elif isinstance(val, bool):
                values.append(str(int(val)))
            elif isinstance(val, str):
                values.append("'{}'".format(val.replace("'", "''")))
            elif isinstance(val, pd.Timestamp):
                values.append(f"'{val}'")
            else:
                values.append(str(val))
        ddl_str += f"INSERT INTO {table_name} VALUES ({', '.join(values)});\n"

    return ddl_str


def get_transpose(df: DataFrame) -> DataFrame:
    return df.T


def serialize_dataframe(df: DataFrame):
    """
    Returns three lightweight string serializations.

    pipe_serialized   Header and cell values joined by ' | '
    token_serialized  TAPAS-style positional tokens
                        <Header, 0, col_idx> and <CellValue, row_idx, col_idx>
    space_serialized  Values joined by spaces with no delimiter
    """
    pipe = " | ".join(df.columns)
    for _, row in df.iterrows():
        pipe += " | ".join(map(str, row.values))

    token = ""
    for col_idx, col_name in enumerate(df.columns):
        token += f"<Header, 0, {col_idx}> {col_name} "
    for row_idx, row in df.iterrows():
        for col_idx, value in enumerate(row.values):
            token += f"<CellValue, {row_idx + 1}, {col_idx}> {value} "

    space = " ".join(df.columns)
    for _, row in df.iterrows():
        space += " ".join(map(str, row.values))

    return pipe, token, space


# ============================================================
# Data loading
# ============================================================

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ============================================================
# Question corpus split
# ============================================================

def build_question_corpus(question_data: dict) -> dict:
    """
    Partitions the flat question dict into per-dataset buckets.
    Unrecognized prefixes are silently skipped.
    """
    corpus = {ds: {} for ds in DATASETS}
    for qid, qinfo in question_data.items():
        for ds in DATASETS:
            if ds in qid:
                corpus[ds][qid] = qinfo
                break
    return corpus


# ============================================================
# Table corpus construction
# ============================================================

def build_table_corpus(tables_data: dict) -> dict:
    """
    Iterates over every dataset and table, applies all serializers,
    and returns the nested table_corpus dict.

    Saves incrementally after each dataset so a crash mid-run does
    not discard already-processed work.
    """
    table_corpus = {}

    for ds, dataset_tables in tables_data.items():
        print(f"{ds}", end=", ", flush=True)
        table_corpus.setdefault(ds, {})

        for table_id, table_df in zip(
            dataset_tables["table_ids_list"],
            dataset_tables["table_dataframe_list"],
        ):
            df = table_df.copy()
            df.columns = sanitize_and_dedupe(df.columns)

            entry = {"df_string": df}

            entry["csv"]           = get_csv(df)
            entry["tsv"]           = get_tsv(df)
            entry["html"]          = get_html(df)
            entry["markdown"]      = get_markdown(df)
            entry["latex"]         = get_latex(df)
            entry["dict"]          = get_dict(df)
            entry["xml"]           = get_xml(df)
            entry["shuffled_rows"] = get_shuffled_rows(df)
            entry["shuffled_cols"] = get_shuffled_cols(df)
            entry["mschema"]       = get_mschema(df)
            entry["macschema"]     = get_macschema(df)
            entry["ddl"]           = get_ddl(df, table_name="Users")
            entry["transpose"]     = get_transpose(df)

            # JSON can fail on exotic dtypes; surface the table_id if so
            try:
                entry["json"] = get_json(df)
            except Exception as exc:
                print(f"\n[WARN] json failed for {table_id}: {exc}")
                entry["json"] = ""

            pipe, token, space = serialize_dataframe(df)
            entry["pipe_serialized"]  = pipe
            entry["token_serialized"] = token
            entry["space_serialized"] = space

            table_corpus[ds][table_id] = entry

        n = len(dataset_tables["table_dataframe_list"])
        print(f"\t{n} tables processed...", flush=True)

        # Incremental save after each dataset
        save_pickle(table_corpus, SAVE_TABLES_PATH)

    return table_corpus


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    
    QUESTION_DATA_PATH = "./data/dataset_all/all_questions_dict_full_wtq_nq_wikisql"
    TABLES_DATA_PATH = "./data/dataset_all/all_tables_full_wtq_nq_wikisql"
    SAVE_QUESTIONS_PATH = "./data/dataset_all/new_all_questions_dict_wtq_nqt_wikisql.pkl"
    SAVE_TABLES_PATH = "./data/dataset_all/all_table_transform_table_corpus.pkl"

    print("Loading question data...")
    question_data = load_pickle(QUESTION_DATA_PATH)

    print("Splitting questions by dataset...")
    question_corpus = build_question_corpus(question_data)
    save_pickle(question_corpus, SAVE_QUESTIONS_PATH)
    print(f"Saved question corpus -> {SAVE_QUESTIONS_PATH}")

    print("Loading table data...")
    tables_data = load_pickle(TABLES_DATA_PATH)

    print("Building table corpus...")
    table_corpus = build_table_corpus(dict(tables_data))
    save_pickle(table_corpus, SAVE_TABLES_PATH)
    print(f"Saved table corpus -> {SAVE_TABLES_PATH}")