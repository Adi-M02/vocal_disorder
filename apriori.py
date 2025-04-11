import sys
import os
import re
from itertools import chain
import pandas as pd
import numpy as np

from analyze_users import prepare_all_users_dataframe

# If using mlxtend for association rule mining:
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# -----------------------------
# 1. LOAD BOTOX DATES CSV
# -----------------------------
botox_df = pd.read_csv("user_botox_dates.csv")
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
# We'll assume you have 'prepare_user_dataframe(username)' already defined, e.g.:
#
# def prepare_user_dataframe(username):
#     # Returns a DataFrame with each post in a row
#     # and columns for each category (after counting occurrences).
#     # Also includes a 'date' column (Datetime) for each post.


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

from vocabulary_evaluation.vocabularies.base_expansion_and_manual import rcpd_terms as term_categories
category_cols = list(term_categories.keys())

# We'll create a transaction for each user * injection_date combination,
# capturing all categories mentioned in a certain time window (e.g. 30 days).

time_window_days = 365
transactions = []

for user, group in merged_df.groupby("user"):
    # get this user's list of injection dates (sorted)
    user_botox_dates = group["botox_dates"].iloc[0]  # same list for all rows for that user
    if not user_botox_dates or len(user_botox_dates) == 0:
        continue  # skip users with no known injection dates
    
    # For each injection date:
    for injection_date in user_botox_dates:
        start_date = injection_date - pd.Timedelta(days=time_window_days)
        # Filter to posts between [start_date, injection_date]
        mask = (group["date"] >= start_date) & (group["date"] <= injection_date)
        window_df = group.loc[mask, category_cols]
        # Summation of category columns across all posts in that window:
        sums = window_df.sum(axis=0)

        # If sum>0 for a category, we treat it as 'mentioned' in this transaction
        mentioned_categories = []
        for col in category_cols:
            if sums[col] > 0:
                mentioned_categories.append(col)

        if len(mentioned_categories) > 0:
            transactions.append(mentioned_categories)
print(f"Total transactions: {len(transactions)}")
if transactions:
    print("Example transaction:", transactions[0])
else:
    print("No transactions generated")

# Now 'transactions' is a list of lists, where each sub-list
# is the set of categories mentioned in the time window prior to an injection.

# -----------------------------
# 4. RUN ASSOCIATION RULE MINING
# -----------------------------
# Convert your transactions into a one-hot DataFrame using TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
basket_df = pd.DataFrame(te_ary, columns=te.columns_)

# Perform Apriori
min_support = 0.01  # adapt threshold as needed
freq_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)

# Generate association rules
rules = association_rules(freq_itemsets, metric="lift", min_threshold=1.0)

# Sort by confidence or lift
rules.sort_values("confidence", ascending=False, inplace=True)

# Print out top rules
print("=== Top Association Rules ===")
print(rules.head(20))

# Optionally, you can filter the rules to focus on certain categories or a minimum lift:
# e.g. rules[ rules['lift'] >= 1.5 ]

# -----------------------------
# 5. INTERPRET RESULTS
# -----------------------------
# 'rules' DataFrame columns:
#   - 'antecedents' : set of items
#   - 'consequents' : set of items
#   - 'support' : frequency of that itemset in transactions
#   - 'confidence' : P(consequents | antecedents)
#   - 'lift' : ratio of observed support to expected support if independent
#
# Inspect the high-confidence rules for clinically relevant patterns.
