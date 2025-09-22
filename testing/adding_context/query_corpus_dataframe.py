import os
import pandas as pd


# ---- LOAD ----
def load_ngram_df(path: str, *, format: str = "parquet") -> pd.DataFrame:
    """
    Load the cached dataframe and re-materialize derived columns so it
    can be passed directly to sample_docs_containing(...).
    """
    if format == "parquet":
        df = pd.read_parquet(path, engine="pyarrow")
    elif format == "feather":
        df = pd.read_feather(path)
    elif format == "pickle":
        df = pd.read_pickle(path)
    else:
        raise ValueError("format must be one of {'parquet','feather','pickle'}")

    # Rebuild derived columns used by your helpers
    df["base_text"] = df["base_tokens"].apply(" ".join)
    df["ngram_text"] = df["ngram_tokens"].apply(" ".join)
    df["ngram_token_set"] = df["ngram_tokens"].apply(set)
    df["base_len"] = df["base_tokens"].apply(len)
    return df