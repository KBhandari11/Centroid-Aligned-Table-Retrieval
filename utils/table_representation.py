from pandas import DataFrame
import pandas as pd
import pandera.pandas as pa
import pandas.io.json
import unicodedata
def get_csv(df: DataFrame,):
    return df.to_csv()

def get_tsv(df: DataFrame, **kwargs):
   
    kwargs.setdefault("sep", "\t")
    return df.to_csv(**kwargs)

def get_html(df: DataFrame):
    return df.to_html()

def get_markdown(df: DataFrame):
    return df.to_markdown()

def get_latex(df: DataFrame):
    return df.to_latex()

def get_dict(df: DataFrame):
    return df.to_dict()

def get_json(df: DataFrame):
    return df.to_json()

def get_xml(df: DataFrame, parser: str = "etree"):
    return df.to_xml(parser=parser, pretty_print=False)

def get_shuffled_rows(df: DataFrame):
    return df.sample(frac=1).reset_index(drop=True)

def get_shuffled_cols(df: DataFrame):
    return df.sample(frac=1, axis=1)



def get_mschema(df: DataFrame):
    #df.columns = _dedup_cols(df.columns)  # ensures df[col] is always a Series
    columns = {}
    for col in df.columns:
        s = df[col]               # Series now
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

    schema = pa.DataFrameSchema(columns)

    return {
        "schema": schema,
        "data": df.to_dict(orient="records"),
    }

def get_macschema(df: DataFrame):
    # Use pandas built-in method to get a metadata aware schema (JSON Table Schema)
    table_schema = pd.io.json.build_table_schema(df)
    
    # Add row data to the schema
    table_schema['data'] = df.to_dict(orient='records')
    
    return table_schema

def get_ddl(df: pd.DataFrame, table_name: str = "Table"):
    dtype_map = {
        'int64': 'INTEGER',
        'float64': 'FLOAT',
        'object': 'TEXT',
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'TIMESTAMP'
    }
    
    # Create DDL for table schema
    ddl = f"CREATE TABLE {table_name} (\n"
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        sql_type = dtype_map.get(dtype_str, "TEXT")
        ddl += f"  {col} {sql_type},\n"
    ddl = ddl.rstrip(',\n') + "\n);\n"
    
    # Create INSERT statements for each row
    insert_statements = ""
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            val = row[col]
            # Format values based on type for SQL syntax
            if pd.isnull(val):
                values.append("NULL")
            elif isinstance(val, str):
                values.append("'{}'".format(val.replace("'", "''")))  
            elif isinstance(val, pd.Timestamp):
                values.append(f"'{val}'")
            elif isinstance(val, bool):
                values.append(str(int(val)))  # Convert bool to 0/1
            else:
                values.append(str(val))
        
        values_str = ", ".join(values)
        insert_statements += f"INSERT INTO {table_name} VALUES ({values_str});\n"
    
    return ddl + insert_statements

def get_transpose(df: DataFrame):
    return df.T


def serialize_dataframe(df: pd.DataFrame) -> str:
    """
    Serialize a pandas DataFrame into multiple formats:
    1. Pipe-separated
    2. Token-based (using <Header,row_idx,col_idx> and <CellValue,row_idx,col_idx>)
    3. JSON string
    Returns the combined string.
    """
    # 1. Pipe-separated serialization
    pipe_serialized = " | ".join(df.columns) 
    for _, row in df.iterrows():
        pipe_serialized += " | ".join(map(str, row.values)) 

    # 2. Token-based serialization
    token_serialized = ""
    # Header tokens
    for col_idx, col_name in enumerate(df.columns):
        token_serialized += f"<Header, 0, {col_idx}> {col_name} "
    # Cell value tokens
    for row_idx, row in df.iterrows():
        for col_idx, value in enumerate(row.values):
            token_serialized += f"<CellValue, {row_idx+1}, {col_idx}> {value} "

    # 1. No separation serialization
    none_serialized = " ".join(df.columns) 
    for _, row in df.iterrows():
        none_serialized += " ".join(map(str, row.values)) 


    return pipe_serialized, token_serialized,none_serialized
import re

def sanitize_and_dedupe(cols, sep="#", empty_name="Column"):
    cleaned = []
    for c in cols:
        s = unicodedata.normalize("NFKC", "" if c is None else str(c))
        s = s.strip()

        if not s:
            s = empty_name
        elif s[0].isdigit():
            s = f"{empty_name} {s}"   # "2020" -> "Column 2020"

        cleaned.append(s)

    seen = {}
    out = []
    for s in cleaned:
        seen[s] = seen.get(s, 0) + 1
        if seen[s] == 1:
            out.append(s)
        else:
            out.append(f"{s} {sep}{seen[s]}")  # "City" -> "City #2"
    return out

