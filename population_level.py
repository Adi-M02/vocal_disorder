import pandas as pd
from tabulate import tabulate
from analyze_users import prepare_all_users_dataframe


# 1. Load Botox CSV & parse dates (optional step, if you need user list or injection info)
botox_df = pd.read_csv("user_botox_dates.csv")
date_cols = [c for c in botox_df.columns if "botox_" in c and "_date" in c]

for col in date_cols:
    botox_df[col] = pd.to_datetime(botox_df[col], format="%m-%d-%y", errors="coerce")

def gather_dates(row):
    user_dates = []
    for c in date_cols:
        if pd.notnull(row[c]):
            user_dates.append(row[c])
    return sorted(user_dates)

botox_df["botox_dates"] = botox_df.apply(gather_dates, axis=1)
all_users = botox_df["user"].unique().tolist()

# 2. Prepare DataFrame with categories
df_all = prepare_all_users_dataframe(all_users)

# 3. Basic population-level stats
from vocabulary_evaluation.vocabularies.base_expansion_and_manual import rcpd_terms as term_categories
category_cols = list(term_categories.keys())
cat_sums = df_all[category_cols].sum()
category_sums_sorted = cat_sums.sort_values(ascending=False)
table_data = [(cat, freq) for cat, freq in category_sums_sorted.items()]
print("\n=== Total Mentions per Category (All Posts) ===")
print(tabulate(table_data, headers=["Category", "Total Mentions"], tablefmt="fancy_grid"))

# --- 3. Unique Users Mentioning Each Category ---
#   i.e., how many distinct users had a sum of >0 for that category?
unique_user_counts = {}
grouped_by_user = df_all.groupby("user")

for cat in category_cols:
    # Sum across all posts by a user -> how many times that user mentioned the category in total
    user_sums = grouped_by_user[cat].sum()
    # Count how many users had >0 mentions
    unique_user_counts[cat] = (user_sums > 0).sum()

# Sort by descending unique user count
unique_user_counts_sorted = dict(
    sorted(unique_user_counts.items(), key=lambda x: x[1], reverse=True)
)

table_data = [(cat, count) for cat, count in unique_user_counts_sorted.items()]
print("\n=== Number of Unique Users who Mention Each Category ===")
print(tabulate(table_data, headers=["Category", "Num. of Users"], tablefmt="fancy_grid"))

# --- 4. Average Mentions per Post ---
#   total mentions of a category / total # of posts
total_posts = len(df_all)
avg_mentions = (cat_sums / total_posts).sort_values(ascending=False)
table_data = [(cat, f"{val:.2f}") for cat, val in avg_mentions.items()]
print("\n=== Average Mentions per Post (All Posts) ===")
print(tabulate(table_data, headers=["Category", "Avg. Mentions/Post"], tablefmt="fancy_grid"))

# --- 5. Pairwise Co-Occurrences (Optional) ---
#   i.e., how often do two categories appear together in the SAME post?
#   This can be useful to see which categories frequently co-occur.

cooccurrences = []
pairs_checked = set()

for i in range(len(category_cols)):
    for j in range(i+1, len(category_cols)):
        cat1 = category_cols[i]
        cat2 = category_cols[j]
        # Count # of posts where cat1>0 AND cat2>0
        count_both = df_all[(df_all[cat1] > 0) & (df_all[cat2] > 0)].shape[0]
        cooccurrences.append((cat1, cat2, count_both))
        pairs_checked.add((cat1, cat2))

# Sort results descending by co-occurrence
cooccurrences_sorted = sorted(cooccurrences, key=lambda x: x[2], reverse=True)

# Print top 10 (or all if you'd like)
top_n = 10
table_data = [
    (c[0], c[1], c[2]) for c in cooccurrences_sorted[:top_n]
]
print(f"\n=== Top {top_n} Category Pairwise Co-Occurrences (Same Post) ===")
print(tabulate(table_data, headers=["Category1", "Category2", "Co-occurrence"], tablefmt="fancy_grid"))

# --- 6. Correlation Matrix (Optional) ---
#   This checks how strongly categories vary together across posts.
#   NOTE: Correlation on count data can be tricky, but it can still give a rough sense.
cat_corr = df_all[category_cols].corr()
print("\n=== Correlation Matrix among Categories ===")
cat_corr = df_all[category_cols].corr()  # Correlation matrix
cat_corr_rounded = cat_corr.round(2)

# Move the row labels ("index") into an actual column for clearer tabulate output
cat_corr_rounded = cat_corr_rounded.reset_index().rename(columns={"index": "Category"})

print("\n=== Correlation Matrix among Categories ===\n")
print(tabulate(
    cat_corr_rounded,          # DataFrame with row labels as a normal column
    headers="keys",            # use the column names as headers
    tablefmt="fancy_grid",     # nice ASCII table format
    showindex=False            # don't print a separate numerical index
))