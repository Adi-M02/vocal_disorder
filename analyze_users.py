import pandas as pd
import numpy as np
import re
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from textblob import TextBlob
from lifelines import KaplanMeierFitter
from datetime import datetime
import query_mongo as query
import text_processing as text
with open('rcpd_terms.json', encoding="utf-8") as _f:
    TERM_CATEGORY_DICT = json.load(_f)
import query_mongo as query

manually_analyzed_users = ['ThinkSuccotash', 'ScratchGolfer1976', 'Mobile-Breakfast-526', 'Wrob88', 'tornteddie', 'AmazingAd5243', ]

term_categories = TERM_CATEGORY_DICT

log = logging.getLogger("bot")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())

# Preprocess terms for accurate bi/trigram matching
def preprocess_terms(category_terms):
    processed = {}
    for term in category_terms:
        term_clean = term.replace('_', ' ').lower()
        processed[term_clean] = re.compile(r'\b' + re.escape(term_clean) + r'\b')
    return processed

# Count occurrences avoiding double counting
def count_category_occurrences(text, processed_terms):
    text_lower = text.lower()
    matched_spans = []
    count = 0
    # Sort terms by length to prioritize bi/trigrams
    for term, pattern in sorted(processed_terms.items(), key=lambda x: -len(x[0])):
        for match in pattern.finditer(text_lower):
            span = match.span()
            if not any(span[0] >= existing_span[0] and span[1] <= existing_span[1] for existing_span in matched_spans):
                matched_spans.append(span)
                count += 1
    return count

# Prepare DataFrame for user analysis
def prepare_user_dataframe(username):
    entries = query.return_user_entries(db_name="reddit", collection_name="noburp_all", user=username, filter_subreddits=["noburp"])
    log.info(f"Retrieved {len(entries)} entries for user {username}")
    df = pd.DataFrame(entries)
    df["date"] = pd.to_datetime(df["created_utc"], unit='s')
    df.sort_values(by="date", inplace=True)

    # Create 'content' field from title + selftext or from body
    def combine_content(row):
        title = row["title"] if "title" in row and pd.notna(row["title"]) else ""
        selftext = row["selftext"] if "selftext" in row and pd.notna(row["selftext"]) else ""
        body = row["body"] if "body" in row and pd.notna(row["body"]) else ""

        content = f"{title} {selftext}".strip()
        return content if content else body.strip()

    df["content"] = df.apply(combine_content, axis=1)

    # Preprocess the content field
    # df["content"] = df["content"].apply(text.preprocess_text)

    # Preprocess category terms and count occurrences
    processed_category_terms = {cat: preprocess_terms(terms) for cat, terms in term_categories.items()}
    for category, processed_terms in processed_category_terms.items():
        df[category] = df["content"].apply(lambda x: count_category_occurrences(x, processed_terms))

    df["month"] = df["date"].dt.to_period("M")
    return df

def prepare_user_dataframe_multi(user_list):
    """
    Prepares a combined DataFrame for a list of users from r/noburp.
    Includes per-post category counts and metadata like month and user ID.
    """
    entries = query.return_multiple_users_entries(
        db_name="reddit",
        collection_name="noburp_all",
        users=user_list,
        filter_subreddits=["noburp"]
    )

    if not entries:
        log.warning("No entries found for the given users.")
        return pd.DataFrame()

    log.info(f"Retrieved {len(entries)} total entries for {len(user_list)} users.")
    df = pd.DataFrame(entries)
    df["date"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df = df[df["date"].notnull()].sort_values(by="date")

    # Ensure necessary columns exist
    for col in ["title", "selftext", "body", "author"]:
        if col not in df.columns:
            df[col] = ""

    # Combine fields into 'content'
    def combine_content(row):
        title = row["title"] if pd.notna(row["title"]) else ""
        selftext = row["selftext"] if pd.notna(row["selftext"]) else ""
        body = row["body"] if pd.notna(row["body"]) else ""
        content = f"{title} {selftext}".strip()
        return content if content else body.strip()

    df["content"] = df.apply(combine_content, axis=1)
    df["user"] = df["author"]

    # Preprocess category terms and count occurrences
    processed_category_terms = {
        cat: preprocess_terms(terms) for cat, terms in term_categories.items()
    }
    for category, processed_terms in processed_category_terms.items():
        df[category] = df["content"].apply(lambda x: count_category_occurrences(x, processed_terms))

    df["month"] = df["date"].dt.to_period("M")
    return df

def prepare_all_users_dataframe(user_list):
    """
    Fetch all user posts from r/noburp in a single MongoDB query,
    then return a combined DataFrame with per-post category counts,
    user column, and post dates.
    """

    entries = query.return_multiple_users_entries(
        db_name="reddit",
        collection_name="noburp_all",
        users=user_list,
        filter_subreddits=["noburp"]
    )

    if not entries:
        print("no entries found")
        return pd.DataFrame()
    print(len(entries))
    df = pd.DataFrame(entries)
    df["date"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df = df[df["date"].notnull()]
    df.sort_values(by="date", inplace=True)

    # Fill missing fields to avoid KeyError
    for col in ["title", "selftext", "body", "author"]:
        if col not in df.columns:
            df[col] = ""

    # Combine title + selftext or fallback to body
    def combine_content(row):
        title = row["title"] if pd.notna(row["title"]) else ""
        selftext = row["selftext"] if pd.notna(row["selftext"]) else ""
        body = row["body"] if pd.notna(row["body"]) else ""
        content = f"{title} {selftext}".strip()
        return content if content else body.strip()

    df["content"] = df.apply(combine_content, axis=1)

    # Add user column
    df["user"] = df["author"]

    # Apply category term processing
    processed_category_terms = {
        cat: preprocess_terms(terms) for cat, terms in term_categories.items()
    }
    for category, processed_terms in processed_category_terms.items():
        df[category] = df["content"].apply(lambda x: count_category_occurrences(x, processed_terms))

    df["month"] = df["date"].dt.to_period("M")
    return df

# Analysis Functions
def plot_category_trends(df, username):
    monthly = df.groupby("month")[list(term_categories.keys())].sum()
    monthly.plot(marker='o', figsize=(12,6), title=f"Category Trends Over Time for {username}")
    plt.xlabel("Month")
    plt.ylabel("Occurrences")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def sentiment_analysis(df):
    df["sentiment"] = df["selftext"].apply(lambda x: TextBlob(x).sentiment.polarity)
    monthly_sentiment = df.groupby("month")["sentiment"].mean()

    monthly_sentiment.plot(marker='o', figsize=(10,5), title="Sentiment Trajectory")
    plt.xlabel("Month")
    plt.ylabel("Average Sentiment Polarity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def category_transition_matrix(df):
    df["dominant_category"] = df[list(term_categories.keys())].idxmax(axis=1)
    transitions = pd.crosstab(df["dominant_category"], df["dominant_category"].shift(-1), normalize='index')

    plt.figure(figsize=(8,6))
    sns.heatmap(transitions, annot=True, cmap="Blues")
    plt.title("Category Transition Matrix")
    plt.ylabel("Current Category")
    plt.xlabel("Next Category")
    plt.show()

def survival_analysis(df, event_category="Self-Treatment & Behavioral Interventions"):
    df["event"] = df[event_category] > 0
    df["timeline"] = (df["date"] - df["date"].iloc[0]).dt.days

    kmf = KaplanMeierFitter()
    kmf.fit(df["timeline"], event_observed=df["event"])

    kmf.plot_survival_function()
    plt.title("Time Until First Mention of Alternative Treatment")
    plt.xlabel("Days")
    plt.ylabel("Survival Probability (No Alternative Treatment Mentioned)")
    plt.tight_layout()
    plt.show()

def category_cooccurrence_network(df):
    import networkx as nx

    G = nx.Graph()
    for _, row in df.iterrows():
        present_categories = [cat for cat in term_categories if row[cat] > 0]
        for cat1 in present_categories:
            for cat2 in present_categories:
                if cat1 != cat2:
                    if G.has_edge(cat1, cat2):
                        G[cat1][cat2]['weight'] += 1
                    else:
                        G.add_edge(cat1, cat2, weight=1)

    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G, k=0.5)
    edges = G.edges(data=True)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u,v):d['weight'] for u,v,d in edges})
    plt.title("Category Co-occurrence Network")
    plt.show()

# Main analysis function
def run_full_user_analysis(username):
    df = prepare_user_dataframe(username)

    plot_category_trends(df, username)
    sentiment_analysis(df)
    # category_transition_matrix(df)
    # survival_analysis(df)
    # category_cooccurrence_network(df)

# # Example usage:
# run_full_user_analysis("AmazingAd5243")

