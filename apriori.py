import sys
import os
import re
from itertools import chain
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json

from analyze_users import prepare_all_users_dataframe

# If using mlxtend for association rule mining:
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# -----------------------------
# 1. LOAD BOTOX DATES CSV
# -----------------------------
botox_df = pd.read_csv("user_botox_dates_fixed.csv")
date_cols = [c for c in botox_df.columns if "botox_" in c and "_date" in c]

# Convert each date col to datetime, ignoring parsing errors
for col in date_cols:
    botox_df[col] = pd.to_datetime(botox_df[col], format="%m-%d-%y", errors="coerce")

# Consolidate the injection dates into a single list per user
def gather_dates(row):
    user_dates = []
    for c in date_cols:
        if pd.notnull(row[c]):
            user_dates.append(row[c])
    return sorted(user_dates)

botox_df["botox_dates"] = botox_df.apply(gather_dates, axis=1)

# The resulting botox_df should look like:
# user | botox_1_date | botox_2_date | ... | botox_dates (list of datetimes)
# -----------------------------
# 2. PREPARE USER DATA
# -----------------------------

# Get the unique list of users from the botox CSV
all_users = botox_df["user"].unique().tolist()

# Retrieve + combine their data
df_all = prepare_all_users_dataframe(all_users)

# Ensure the 'date' column is datetime
df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")

# Merge with botox_df to attach the 'botox_dates' list to each user
merged_df = df_all.merge(botox_df[["user", "botox_dates"]], on="user", how="left")

# -----------------------------
# 3. BUILD TRANSACTIONS
# -----------------------------
# We assume each row in merged_df has columns for categories:
# e.g. 'comorb_anxiety', 'comorb_depression', 'comorb_ibs', ...
# or any names you used (like categories in your code snippet).
#
# For demonstration, let's say these columns hold integer counts
# of how many times that category was mentioned in a post.
# We'll define which columns represent categories:

# create a transaction for each user * injection_date combination,
# capturing all categories mentioned in a certain time window

# import vacabulary
from vocabulary_evaluation.vocabularies.base_expansion_and_manual import rcpd_terms as term_categories
category_cols = list(term_categories.keys())

direction = "after"
time_window_days = 30
transactions = []

for user, group in merged_df.groupby("user"):
    user_botox_dates = group["botox_dates"].iloc[0]
    if not user_botox_dates or len(user_botox_dates) == 0:
        continue

    for injection_date in user_botox_dates:
        if direction == "before":
            start_date = injection_date - pd.Timedelta(days=time_window_days)
            end_date = injection_date
        elif direction == "after":
            start_date = injection_date
            end_date = injection_date + pd.Timedelta(days=time_window_days)
        else:
            raise ValueError("Invalid direction: must be 'before' or 'after'")

        # Filter to posts in the selected time window
        mask = (group["date"] >= start_date) & (group["date"] <= end_date)
        window_df = group.loc[mask, category_cols]

        # Count category mentions
        sums = window_df.sum(axis=0)
        mentioned_categories = [col for col in category_cols if sums[col] > 0]

        if mentioned_categories:
            transactions.append(mentioned_categories)
print(f"Total transactions: {len(transactions)}")

# Now 'transactions' is a list of lists, where each sub-list
# is the set of categories mentioned in the time window prior to an injection.

# -----------------------------
# 4. RUN ASSOCIATION RULE MINING
# -----------------------------

# Run metadata
min_support = 0.01
min_lift = 1.0
today = datetime.today()
folder_date = today.strftime("%m_%d")

# Folder and run tag
output_folder = Path(f"apriori_output_{folder_date}")
output_folder.mkdir(parents=True, exist_ok=True)

run_tag = f"{direction}_window{time_window_days}d_support{min_support}_lift{min_lift}"

# save vocab used in analysis to folder 
vocab_path = output_folder / f"used_vocabulary_{run_tag}.json"
vocab_serializable = {
    category: sorted(list(map(str, terms)))
    for category, terms in term_categories.items()
}

with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(vocab_serializable, f, indent=2, ensure_ascii=False)
    
# One-hot encode transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
basket_df = pd.DataFrame(te_ary, columns=te.columns_)

# Run Apriori
freq_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)
freq_itemsets_sorted = freq_itemsets.sort_values(by="support", ascending=False).reset_index(drop=True)

# Save frequent itemsets
freq_itemsets_path = output_folder / f"frequent_itemsets_{run_tag}.csv"
freq_itemsets_sorted.to_csv(freq_itemsets_path, index=False)

# Generate rules
rules = association_rules(freq_itemsets, metric="lift", min_threshold=min_lift)
rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
rules_sorted = rules.sort_values(by="confidence", ascending=False).reset_index(drop=True)

# Save rules
rules_path = output_folder / f"association_rules_{run_tag}.csv"
rules_sorted.to_csv(rules_path, index=False)

# Confirm
print(f"\n✅ Saved to: {output_folder}")
print(f"📄 Frequent itemsets: {freq_itemsets_path.name}")
print(f"📄 Association rules: {rules_path.name}")
print("\n=== Top Association Rules ===")
print(rules_sorted.head(10)[["antecedents", "consequents", "support", "confidence", "lift"]])