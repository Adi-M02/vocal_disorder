import pandas as pd

df = pd.read_parquet("base_ngram_cache_with_details.parquet")
df.to_csv("base_ngram_cache_with_details.csv")