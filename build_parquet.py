if __name__ == "__main__":
    df = build_ngram_df("testing/ngram_evals_test_no_digits/4")
    cache_path = "base_ngram_cache_with_details.parquet"
    save_ngram_df(df, cache_path, format="parquet")