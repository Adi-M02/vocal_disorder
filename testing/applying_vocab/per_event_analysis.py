#!/usr/bin/env python3
"""
per_event_vocab_analysis.py

per-event vocabulary analysis for Reddit-like corpora.
- Event day INCLUDED in post: post window uses ts >= t0 (timestamp-based).
- Runs multiple windows in one run (default: 180,90,30,5 days).
- Overlap handling across a user's multiple events (default: clip at midpoint).

Computes per window:
1) Per-term pre vs post: counts, proportions, lift, OR(+0.5), log-odds, z, p, q (BH).
2) Windowed time series: per-day/per-week rates with Wilson CIs.
3) Paired within-user: McNemar's test (exact) + direction.
4) Association rules: (pre terms) ⇒ (post terms), with support, confidence, lift.
5) Exports: CSVs (terms, optional categories), PNG plots, HTML summary per window,
   plus an index.html linking all windows.

Inputs (parquet):
    - ngram_text (str)  — already vocab-formatted (underscored n-grams)
    - created_utc (int64, epoch seconds UTC)
    - author (str)
    - ngram_tokens (object)       [optional]
    - ngram_token_set (object)    [optional]
    - ngram_len (int64)           [unused here]

Botox CSV: user, botox_1_date,... (MM-DD-YY). Parsed as America/Chicago @ 12:00, converted to UTC.
Vocabulary JSON: ["term_a","term_b",...]
Optional term→category JSON: {"term_a":"Category", ...}

CLI example:
python per_event_vocab_analysis.py \
  --parquet path/to/posts.parquet \
  --botox_csv path/to/botox_dates.csv \
  --vocab_json path/to/vocab.json \
  --outdir outputs/rcpd_event_multi \
  --windows 180,90,30,5 \
  --bin_days 7 \
  --min_support 0.05 \
  --overlap_policy clip_midpoint
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy.stats import binomtest
import matplotlib.pyplot as plt

from datetime import datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # Fallback if unavailable


# ------------------------- I/O helpers -------------------------

def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    (outdir / "csv").mkdir(parents=True, exist_ok=True)


def parse_botox_dates(csv_path: Path,
                      local_tz: str = "America/Chicago") -> pd.DataFrame:
    """
    Robust parser for your botox-date CSV.

    Accepts headers with BOM/whitespace/case differences and looks for a user-like
    column in ['user','username','author'] (case-insensitive). Date columns are any
    that start with 'botox' and usually end with '_date'. Dates like '6-28-23' are
    interpreted in local time at 12:00 and converted to UTC.

    Returns: DataFrame with columns ['user','event_time_utc'] (UTC tz-aware).
    """
    # Handle BOM and weird encodings
    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")

    # Normalize column names (strip + keep original for selection)
    orig_cols = list(df.columns)
    df.columns = [("" if c is None else str(c)).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    # Find the user column (flexible)
    user_key = None
    for cand in ("user", "username", "author"):
        if cand in lower_map:
            user_key = lower_map[cand]
            break
    if user_key is None:
        # fallback: assume first column is user
        user_key = df.columns[0]
        # If you want to fail hard instead, uncomment:
        # raise ValueError(f"No 'user' column found in CSV header: {orig_cols}")

    # Find date columns
    date_cols = [c for c in df.columns if c.lower().startswith("botox")]
    if not date_cols:
        raise ValueError(f"No botox_* date columns found. Got columns: {orig_cols}")

    rows = []
    tzinfo = ZoneInfo(local_tz) if ZoneInfo is not None else None

    for _, row in df.iterrows():
        user = str(row[user_key]).strip()
        if not user or user.lower() == "nan":
            continue
        for dc in date_cols:
            v = str(row.get(dc, "")).strip()
            if not v or v.lower() == "nan":
                continue

            # Parse as date only, then set local noon
            # Accepts 6-28-23, 6/28/23, 2023-06-28, etc.
            dt = pd.to_datetime(v, errors="coerce", dayfirst=False, infer_datetime_format=True)
            if pd.isna(dt):
                continue
            dt = dt.to_pydatetime()
            # strip time part if any
            dt = datetime(dt.year, dt.month, dt.day, 12, 0, 0, 0)

            if tzinfo is not None:
                dt = dt.replace(tzinfo=tzinfo).astimezone(timezone.utc)
            else:
                dt = dt.replace(tzinfo=timezone.utc)

            rows.append({"user": user, "event_time_utc": pd.Timestamp(dt)})

    events = pd.DataFrame(rows).drop_duplicates()
    if events.empty:
        raise ValueError("Parsed 0 events from CSV — check headers/dates formatting.")

    events = events.sort_values(["user", "event_time_utc"]).reset_index(drop=True)
    return events


def _coerce_tokens_to_set(x) -> Set[str]:
    """
    Normalize 'object' shapes into a Python set[str].
    Accepts set, list/tuple/np.ndarray of strings, JSON-ish strings, or space-separated string.
    """
    if isinstance(x, set):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return set()
    if isinstance(x, (list, tuple, np.ndarray)):
        return set(map(str, x))
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return set()
        # Try JSON list like '["a","b"]'
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                val = json.loads(s)
                if isinstance(val, (list, tuple, set, np.ndarray)):
                    return set(map(str, val))
            except Exception:
                pass
        # fallback: space-separated
        return set(s.split())
    try:
        return set(x)
    except Exception:
        return set()


def load_posts(parquet_path: Path) -> pd.DataFrame:
    """
    Loads posts parquet and returns df with:
    author (str), ts (pd.Timestamp UTC), tokens (set[str])

    Prefers 'ngram_token_set', else 'ngram_tokens', else split 'ngram_text'.
    """
    df = pd.read_parquet(parquet_path)
    required = {"author", "created_utc", "ngram_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing required columns: {missing}")

    df = df.copy()
    df["ts"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)

    token_source = None
    if "ngram_token_set" in df.columns:
        token_source = "ngram_token_set"
    elif "ngram_tokens" in df.columns:
        token_source = "ngram_tokens"

    if token_source:
        df["tokens"] = df[token_source].apply(_coerce_tokens_to_set)
    else:
        df["tokens"] = df["ngram_text"].fillna("").astype(str).str.split().apply(set)

    return df[["author", "ts", "tokens"]]


# ------------------------- Stats helpers -------------------------

def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = np.empty_like(p)
    ranked[order] = np.arange(1, n + 1)
    q = p * n / ranked
    q_sorted = q[order]
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_adj = np.empty_like(q)
    q_adj[order] = q_sorted
    return np.clip(q_adj, 0, 1)


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    half_width = (z / denom) * math.sqrt((p*(1-p)/n) + (z**2/(4*n**2)))
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def two_prop_z(x1: int, n1: int, x2: int, n2: int) -> Tuple[float, float]:
    if n1 == 0 or n2 == 0:
        return (np.nan, np.nan)
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    denom = math.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
    if denom == 0:
        return (np.nan, np.nan)
    z = (p1 - p2) / denom
    from math import erf, sqrt
    p = 2 * (1 - 0.5*(1 + erf(abs(z)/sqrt(2))))
    return (z, p)


# ------------------------- Overlap handling -------------------------

def build_event_windows_for_user(
    times: List[pd.Timestamp],
    pre_days: int,
    post_days: int,
    policy: str = "clip_midpoint"
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Given a sorted list of event times for one user, return a list of
    (pre_start, pre_end, post_start, post_end) per event.

    Event day included in post: post_start = t_i (inclusive).
    Default policy 'clip_midpoint': clip left/right ends to halfway between adjacent events.
    Other policies:
      - 'allow': no clipping (windows can overlap; docs can count for multiple events)
      - 'drop_later': if an event is closer than window, drop this event entirely
    """
    pre_delta = timedelta(days=pre_days)
    post_delta = timedelta(days=post_days)
    n = len(times)
    out = []

    if policy == "allow":
        for t in times:
            pre_start = t - pre_delta
            pre_end = t  # exclusive
            post_start = t  # inclusive (include event moment)
            post_end = t + post_delta
            out.append((pre_start, pre_end, post_start, post_end))
        return out

    if policy == "drop_later":
        keep = [True] * n
        for i in range(1, n):
            # if windows would overlap badly, drop later
            prev = times[i-1]
            curr = times[i]
            if (curr - prev) < max(pre_delta, post_delta):
                keep[i] = False
        for i, t in enumerate(times):
            if not keep[i]:
                out.append((None, None, None, None))
                continue
            pre_start = t - pre_delta
            pre_end = t
            post_start = t
            post_end = t + post_delta
            out.append((pre_start, pre_end, post_start, post_end))
        return out

    # clip_midpoint (default)
    mids = []
    for i in range(n - 1):
        mids.append(times[i] + (times[i+1] - times[i]) / 2)

    for i, t in enumerate(times):
        left_bound = t - pre_delta
        right_bound = t + post_delta

        if i > 0:
            left_bound = max(left_bound, mids[i-1])  # clip left by midpoint with previous
        if i < n - 1:
            right_bound = min(right_bound, mids[i])  # clip right by midpoint with next

        pre_start = min(t, left_bound)  # ensure not after t
        pre_end = t                     # exclusive
        post_start = t                  # inclusive
        post_end = max(t, right_bound)  # ensure not before t

        # If clipping collapses a side, windows may be empty; handled later
        out.append((pre_start, pre_end, post_start, post_end))

    return out


# ------------------------- Core aggregation -------------------------

def build_author_index(posts: pd.DataFrame) -> Dict[str, np.ndarray]:
    by_author = defaultdict(list)
    for i, a in enumerate(posts["author"].values):
        by_author[str(a)].append(i)
    return {a: np.asarray(ix, dtype=np.int32) for a, ix in by_author.items()}


def collect_perevent_counts(
    posts: pd.DataFrame,
    events: pd.DataFrame,
    vocab: Set[str],
    pre_days: int,
    post_days: int,
    term_to_cat: Optional[Dict[str, str]] = None,
    bin_days: int = 7,
    overlap_policy: str = "clip_midpoint",
):
    """
    Aggregates across events (after per-user overlap handling):
      - doc totals pre/post
      - term doc counts pre/post
      - McNemar b,c counts per term (any-term-in-window unions)
      - time-series denominators and term/bin numerators
      - optional category versions
    """
    by_author_idx = build_author_index(posts)
    bin_days = int(bin_days)

    docs_pre_total = 0
    docs_post_total = 0
    term_pre_count = Counter()
    term_post_count = Counter()

    mcn_b = Counter()  # pre=1, post=0
    mcn_c = Counter()  # pre=0, post=1

    denom_by_bin = Counter()
    term_bin_counts: Dict[str, Counter] = {t: Counter() for t in vocab}

    categories = sorted(set(term_to_cat.values())) if term_to_cat else []
    term_to_cat_map = {t: c for t, c in (term_to_cat or {}).items() if t in vocab}
    cat_bin_counts: Dict[str, Counter] = {c: Counter() for c in categories}
    cat_pre_count = Counter()
    cat_post_count = Counter()

    # group events by user and precompute windows respecting overlap policy
    events_by_user = {u: g.sort_values("event_time_utc") for u, g in events.groupby("user", dropna=True)}
    for user, g in events_by_user.items():
        t_list = list(g["event_time_utc"].to_list())
        windows = build_event_windows_for_user(t_list, pre_days, post_days, policy=overlap_policy)

        if user not in by_author_idx:
            continue
        rows = by_author_idx[user]
        user_df = posts.iloc[rows]

        for (t0, win) in zip(t_list, windows):
            pre_start, pre_end, post_start, post_end = win
            if pre_start is None:  # dropped
                continue
            # PRE: [pre_start, pre_end) (event moment excluded)
            pre_docs = user_df[(user_df["ts"] >= pre_start) & (user_df["ts"] < pre_end)]
            # POST: [post_start, post_end] (event moment included)
            post_docs = user_df[(user_df["ts"] >= post_start) & (user_df["ts"] <= post_end)]

            docs_pre_total += len(pre_docs)
            docs_post_total += len(post_docs)

            pre_union_terms: Set[str] = set()
            pre_union_cats: Set[str] = set()
            for _, d in pre_docs.iterrows():
                present_terms = d["tokens"] & vocab
                if present_terms:
                    pre_union_terms.update(present_terms)
                    for t in present_terms:
                        term_pre_count[t] += 1
                # time series binning: relative to t0 (in days)
                rel_days = int(math.floor((d["ts"] - t0).total_seconds() / 86400))
                bin_idx = (rel_days // bin_days) * bin_days
                denom_by_bin[bin_idx] += 1
                for t in present_terms:
                    term_bin_counts[t][bin_idx] += 1
                if term_to_cat_map and present_terms:
                    for c in {term_to_cat_map[t] for t in present_terms if t in term_to_cat_map}:
                        cat_bin_counts[c][bin_idx] += 1
                        pre_union_cats.add(c)

            post_union_terms: Set[str] = set()
            post_union_cats: Set[str] = set()
            for _, d in post_docs.iterrows():
                present_terms = d["tokens"] & vocab
                if present_terms:
                    post_union_terms.update(present_terms)
                    for t in present_terms:
                        term_post_count[t] += 1
                rel_days = int(math.floor((d["ts"] - t0).total_seconds() / 86400))
                bin_idx = (rel_days // bin_days) * bin_days
                denom_by_bin[bin_idx] += 1
                for t in present_terms:
                    term_bin_counts[t][bin_idx] += 1
                if term_to_cat_map and present_terms:
                    for c in {term_to_cat_map[t] for t in present_terms if t in term_to_cat_map}:
                        cat_bin_counts[c][bin_idx] += 1
                        post_union_cats.add(c)

            # McNemar discordant pairs
            b_terms = pre_union_terms - post_union_terms
            c_terms = post_union_terms - pre_union_terms
            for t in b_terms:
                mcn_b[t] += 1
            for t in c_terms:
                mcn_c[t] += 1

            # Category presence (any in window)
            if term_to_cat_map:
                for c in pre_union_cats:
                    cat_pre_count[c] += 1
                for c in post_union_cats:
                    cat_post_count[c] += 1

    return {
        "docs_pre_total": docs_pre_total,
        "docs_post_total": docs_post_total,
        "term_pre_count": term_pre_count,
        "term_post_count": term_post_count,
        "mcn_b": mcn_b,
        "mcn_c": mcn_c,
        "denom_by_bin": denom_by_bin,
        "term_bin_counts": term_bin_counts,
        "categories": categories,
        "cat_pre_count": cat_pre_count,
        "cat_post_count": cat_post_count,
        "cat_bin_counts": cat_bin_counts,
    }


# ------------------------- Computations -------------------------

def compute_per_term_stats(agg, vocab: List[str]) -> pd.DataFrame:
    docs_pre = agg["docs_pre_total"]
    docs_post = agg["docs_post_total"]
    term_pre = agg["term_pre_count"]
    term_post = agg["term_post_count"]

    rows = []
    for t in vocab:
        x_pre = term_pre.get(t, 0)
        x_post = term_post.get(t, 0)
        n_pre = docs_pre
        n_post = docs_post

        p_pre = (x_pre / n_pre) if n_pre > 0 else np.nan
        p_post = (x_post / n_post) if n_post > 0 else np.nan
        lift = (p_post / p_pre) if (p_pre not in [0, np.nan] and not np.isnan(p_pre)) else np.inf

        # Haldane–Anscombe 0.5 correction
        a = x_post + 0.5
        b = (n_post - x_post) + 0.5
        c = x_pre + 0.5
        d = (n_pre - x_pre) + 0.5
        OR = (a * d) / (b * c)
        log_odds = math.log(OR)

        z, p = two_prop_z(x_post, n_post, x_pre, n_pre)

        rows.append({
            "term": t,
            "x_pre": x_pre, "n_pre": n_pre, "p_pre": p_pre,
            "x_post": x_post, "n_post": n_post, "p_post": p_post,
            "lift": lift,
            "odds_ratio": OR,
            "log_odds": log_odds,
            "z": z, "p": p
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["q_bh"] = benjamini_hochberg(df["p"].fillna(1.0).values)
    return df.sort_values("p", na_position="last")


def compute_mcnemar(agg, vocab: List[str]) -> pd.DataFrame:
    b_counts = agg["mcn_b"]
    c_counts = agg["mcn_c"]
    rows = []
    for t in vocab:
        b = b_counts.get(t, 0)
        c = c_counts.get(t, 0)
        n = b + c
        if n == 0:
            p = np.nan
        else:
            res = binomtest(k=min(b, c), n=n, p=0.5, alternative="two-sided")
            p = res.pvalue
        direction = "increase" if c > b else ("decrease" if b > c else "no_change")
        rows.append({"term": t, "b_pre1_post0": b, "c_pre0_post1": c, "mcnemar_p": p, "direction": direction})
    df = pd.DataFrame(rows).sort_values("mcnemar_p", na_position="last")
    if len(df) > 0:
        df["mcnemar_q_bh"] = benjamini_hochberg(df["mcnemar_p"].fillna(1.0).values)
    return df


def compute_time_series(agg, vocab: List[str], bin_days: int) -> pd.DataFrame:
    denom_by_bin = agg["denom_by_bin"]
    term_bin_counts = agg["term_bin_counts"]
    bins = sorted(denom_by_bin.keys())
    rows = []
    for t in vocab:
        counts = term_bin_counts[t]
        for b in bins:
            n = denom_by_bin.get(b, 0)
            k = counts.get(b, 0)
            rate = (k / n) if n > 0 else np.nan
            lo, hi = wilson_ci(k, n)
            rows.append({
                "bin_start_day": b,
                "bin_days": bin_days,
                "term": t,
                "k": k, "n": n, "rate": rate, "ci_lo": lo, "ci_hi": hi
            })
    return pd.DataFrame(rows)


def compute_categories(agg) -> Optional[pd.DataFrame]:
    cats = agg["categories"]
    if not cats:
        return None
    n_pre = agg["docs_pre_total"]
    n_post = agg["docs_post_total"]
    pre = agg["cat_pre_count"]
    post = agg["cat_post_count"]
    rows = []
    for c in cats:
        x_pre = pre.get(c, 0)
        x_post = post.get(c, 0)
        p_pre = x_pre / n_pre if n_pre > 0 else np.nan
        p_post = x_post / n_post if n_post > 0 else np.nan
        lift = (p_post / p_pre) if (p_pre not in [0, np.nan] and not np.isnan(p_pre)) else np.inf
        rows.append({"category": c, "x_pre": x_pre, "n_pre": n_pre, "p_pre": p_pre,
                     "x_post": x_post, "n_post": n_post, "p_post": p_post, "lift": lift})
    return pd.DataFrame(rows).sort_values("lift", ascending=False)


def compute_association_rules(
    events: pd.DataFrame,
    posts: pd.DataFrame,
    vocab_set: Set[str],
    pre_days: int,
    post_days: int,
    min_support: float = 0.05,
    overlap_policy: str = "clip_midpoint",
) -> pd.DataFrame:
    """
    Transaction per event:
      pre_set = {terms present in any pre doc}, post_set = {terms present in any post doc}.
    Support computed across events after applying overlap policy.
    """
    by_author = {u: g.sort_values("event_time_utc") for u, g in events.groupby("user", dropna=True)}
    by_author_idx = defaultdict(list)
    for i, a in enumerate(posts["author"].values):
        by_author_idx[str(a)].append(i)

    pre_delta = timedelta(days=pre_days)
    post_delta = timedelta(days=post_days)

    N = 0
    count_X = Counter()
    count_Y = Counter()
    count_XY = Counter()

    for user, g in by_author.items():
        if user not in by_author_idx:
            continue
        rows = np.asarray(by_author_idx[user], dtype=np.int32)
        user_df = posts.iloc[rows]
        times = list(g["event_time_utc"].to_list())
        windows = build_event_windows_for_user(times, pre_days, post_days, policy=overlap_policy)

        for (t0, win) in zip(times, windows):
            pre_start, pre_end, post_start, post_end = win
            if pre_start is None:
                continue

            pre_docs = user_df[(user_df["ts"] >= pre_start) & (user_df["ts"] < pre_end)]
            post_docs = user_df[(user_df["ts"] >= post_start) & (user_df["ts"] <= post_end)]

            pre_set = set()
            for _, d in pre_docs.iterrows():
                pre_set |= (d["tokens"] & vocab_set)
            post_set = set()
            for _, d in post_docs.iterrows():
                post_set |= (d["tokens"] & vocab_set)

            if len(pre_set) == 0 and len(post_set) == 0:
                continue

            N += 1
            for X in pre_set:
                count_X[X] += 1
            for Y in post_set:
                count_Y[Y] += 1
            for X in pre_set:
                for Y in post_set:
                    count_XY[(X, Y)] += 1

    rows = []
    if N == 0:
        return pd.DataFrame(rows)

    for (X, Y), c_xy in count_XY.items():
        sup = c_xy / N
        if sup < min_support:
            continue
        supX = count_X[X] / N if count_X[X] > 0 else 0.0
        supY = count_Y[Y] / N if count_Y[Y] > 0 else 0.0
        conf = sup / supX if supX > 0 else np.nan
        lift = (sup / (supX * supY)) if (supX > 0 and supY > 0) else np.nan
        rows.append({
            "antecedent_pre": X,
            "consequent_post": Y,
            "support": sup,
            "support_X": supX,
            "support_Y": supY,
            "confidence": conf,
            "lift": lift,
            "count_XY": c_xy,
            "N_events": N
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["lift", "confidence", "support"], ascending=False, na_position="last")


# ------------------------- Plotting & report -------------------------

def plot_top_lift(df_terms: pd.DataFrame, outdir: Path, topk: int = 20):
    df = df_terms.replace([np.inf, -np.inf], np.nan).dropna(subset=["lift"])
    if df.empty:
        return None
    df["log2_lift"] = np.log2(df["lift"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["log2_lift"])
    if len(df) == 0:
        return None
    top = df.reindex(df["log2_lift"].abs().sort_values(ascending=False).index).head(topk)
    plt.figure(figsize=(10, 6))
    plt.barh(top["term"], top["log2_lift"])
    plt.axvline(0, linestyle="--")
    plt.gca().invert_yaxis()
    plt.xlabel("log2(lift)  (post vs pre)")
    plt.title(f"Top {topk} terms by |log2(lift)|")
    path = outdir / "plots" / "top_terms_log2lift.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_time_series(df_ts: pd.DataFrame, outdir: Path, terms: List[str], title: str):
    if df_ts.empty or not terms:
        return None
    plt.figure(figsize=(11, 6))
    for t in terms:
        sub = df_ts[df_ts["term"] == t].sort_values("bin_start_day")
        if sub.empty:
            continue
        x = sub["bin_start_day"].values
        y = sub["rate"].values
        lo = sub["ci_lo"].values
        hi = sub["ci_hi"].values
        plt.plot(x, y, label=t)
        plt.fill_between(x, lo, hi, alpha=0.2)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Relative day (bin start)")
    plt.ylabel("Rate (docs with term)")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    path = outdir / "plots" / "time_series.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def write_html_summary(
    outdir: Path,
    per_term_csv: Path,
    mcnemar_csv: Path,
    ts_csv: Path,
    assoc_csv: Path,
    cat_csv: Optional[Path],
    top_plot: Optional[Path],
    ts_plot: Optional[Path],
    window_days: int,
    args: argparse.Namespace
):
    html = []
    html.append("<html><head><meta charset='utf-8'><title>per-Event Vocabulary Summary</title></head><body>")
    html.append(f"<h1>per-Event Vocabulary Summary (window = ±{window_days} days)</h1>")
    html.append("<h3>Inputs</h3>")
    html.append("<ul>")
    html.append(f"<li>Parquet: {args.parquet}</li>")
    html.append(f"<li>Botox CSV: {args.botox_csv}</li>")
    html.append(f"<li>Vocab JSON: {args.vocab_json}</li>")
    if args.cat_json:
        html.append(f"<li>Category JSON: {args.cat_json}</li>")
    html.append(f"<li>Event day included in post: ts ≥ t0</li>")
    html.append(f"<li>Bin size: {args.bin_days} days; Overlap policy: {args.overlap_policy}</li>")
    html.append("</ul>")

    html.append("<h3>Downloads</h3><ul>")
    html.append(f"<li><a href='../{outdir.name}/csv/{per_term_csv.name}'>Per-term pre/post stats</a></li>")
    html.append(f"<li><a href='../{outdir.name}/csv/{mcnemar_csv.name}'>McNemar paired test</a></li>")
    html.append(f"<li><a href='../{outdir.name}/csv/{ts_csv.name}'>Time series (bins)</a></li>")
    html.append(f"<li><a href='../{outdir.name}/csv/{assoc_csv.name}'>Association rules (pre⇒post)</a></li>")
    if cat_csv is not None:
        html.append(f"<li><a href='../{outdir.name}/csv/{cat_csv.name}'>Category pre/post stats</a></li>")
    html.append("</ul>")

    if top_plot:
        html.append("<h3>Top movers (by |log2(lift)|)</h3>")
        html.append(f"<img src='../{outdir.name}/plots/{Path(top_plot).name}' style='max-width: 100%;'>")
    if ts_plot:
        html.append("<h3>Time series (top movers)</h3>")
        html.append(f"<img src='../{outdir.name}/plots/{Path(ts_plot).name}' style='max-width: 100%;'>")

    html.append("</body></html>")
    (outdir / "summary.html").write_text("\n".join(html), encoding="utf-8")


# ------------------------- Main -------------------------

def run_for_window(
    posts: pd.DataFrame,
    events: pd.DataFrame,
    vocab_list: List[str],
    term_to_cat: Optional[Dict[str, str]],
    base_outdir: Path,
    window_days: int,
    bin_days: int,
    min_support: float,
    overlap_policy: str,
    topk_plot: int,
    args: argparse.Namespace,
):
    outdir = base_outdir / f"w{window_days}"
    ensure_outdir(outdir)

    vocab_set = set(vocab_list)

    agg = collect_perevent_counts(
        posts=posts, events=events, vocab=vocab_set,
        pre_days=window_days, post_days=window_days,
        term_to_cat=term_to_cat, bin_days=bin_days,
        overlap_policy=overlap_policy
    )

    # Per-term stats
    df_terms = compute_per_term_stats(agg, vocab_list)
    per_term_csv = outdir / "csv" / "per_term_prepost_stats.csv"
    df_terms.to_csv(per_term_csv, index=False)

    # McNemar
    df_mcn = compute_mcnemar(agg, vocab_list)
    mcn_csv = outdir / "csv" / "mcnemar_paired.csv"
    df_mcn.to_csv(mcn_csv, index=False)

    # Time series
    df_ts = compute_time_series(agg, vocab_list, bin_days)
    ts_csv = outdir / "csv" / "time_series_bins.csv"
    df_ts.to_csv(ts_csv, index=False)

    # Association rules
    df_rules = compute_association_rules(
        events=events, posts=posts, vocab_set=vocab_set,
        pre_days=window_days, post_days=window_days,
        min_support=min_support, overlap_policy=overlap_policy
    )
    assoc_csv = outdir / "csv" / "assoc_rules_pre_to_post.csv"
    df_rules.to_csv(assoc_csv, index=False)

    # Categories
    cat_csv_path = None
    if term_to_cat:
        df_cats = compute_categories(agg)
        if df_cats is not None:
            cat_csv_path = outdir / "csv" / "categories_prepost_stats.csv"
            df_cats.to_csv(cat_csv_path, index=False)

    # Plots
    top_plot = plot_top_lift(df_terms, outdir, topk=topk_plot)
    ts_terms = []
    if "log_odds" in df_terms.columns and len(df_terms) > 0:
        tmp = df_terms.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_odds"])
        tmp = tmp.reindex(tmp["log_odds"].abs().sort_values(ascending=False).index).head(min(topk_plot, 8))
        ts_terms = tmp["term"].tolist()
    ts_plot = plot_time_series(df_ts[df_ts["term"].isin(ts_terms)], outdir, ts_terms, f"Rates around event (±{window_days}d)")

    # Summary for this window
    write_html_summary(outdir, per_term_csv, mcn_csv, ts_csv, assoc_csv,
                       cat_csv_path, top_plot, ts_plot, window_days, args)
    return outdir


def main():
    ap = argparse.ArgumentParser(description="per-event vocabulary analysis (event day included in post)")
    ap.add_argument("--parquet", default='base_ngram_cache_with_details.parquet', type=Path)
    ap.add_argument("--botox_csv", default='annotated_users/user_botox_dates_fixed.csv', type=Path)
    ap.add_argument("--vocab_json", required=True, type=Path)
    ap.add_argument("--cat_json", type=Path, default=None)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--windows", type=str, default="180,90,30,5", help="Comma-separated day windows to run, e.g., 180,90,30,5")
    ap.add_argument("--bin_days", type=int, default=7)
    ap.add_argument("--min_support", type=float, default=0.05)
    ap.add_argument("--topk_plot", type=int, default=20)
    ap.add_argument("--overlap_policy", type=str, default="clip_midpoint",
                    choices=["clip_midpoint", "allow", "drop_later"],
                    help="Handling of events closer than the window for the same user")
    args = ap.parse_args()

    # Prepare base outdir + an index page
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Loading posts…")
    posts = load_posts(args.parquet)

    print("Loading events…")
    events = parse_botox_dates(args.botox_csv)
    if events.empty:
        raise SystemExit("No valid botox events parsed. Check date formats.")
    print(f"Parsed {len(events)} events for {events['user'].nunique()} users.")

    print("Loading vocabulary…")
    vocab_list: List[str] = json.loads(Path(args.vocab_json).read_text(encoding="utf-8"))

    term_to_cat = None
    if args.cat_json and args.cat_json.exists():
        term_to_cat = json.loads(args.cat_json.read_text(encoding="utf-8"))

    window_sizes = [int(s.strip()) for s in args.windows.split(",") if s.strip()]
    html_links = []

    for w in window_sizes:
        print(f"\n=== Running window ±{w} days (event day included in post) ===")
        outdir_w = run_for_window(
            posts=posts,
            events=events,
            vocab_list=vocab_list,
            term_to_cat=term_to_cat,
            base_outdir=args.outdir,
            window_days=w,
            bin_days=args.bin_days,
            min_support=args.min_support,
            overlap_policy=args.overlap_policy,
            topk_plot=args.topk_plot,
            args=args,
        )
        html_links.append((w, outdir_w / "summary.html"))

    # Write a simple index HTML linking all windows
    index_html = ["<html><head><meta charset='utf-8'><title>per-Event Analyses</title></head><body>"]
    index_html.append("<h1>per-Event Analyses (multiple windows)</h1>")
    index_html.append(f"<p>Parquet: {args.parquet}<br>Botox CSV: {args.botox_csv}<br>Vocab: {args.vocab_json}</p>")
    index_html.append(f"<p>Overlap policy: <b>{args.overlap_policy}</b> (event day included in post)</p>")
    index_html.append("<ul>")
    for w, pth in html_links:
        rel = pth.relative_to(args.outdir)
        index_html.append(f"<li><a href='{rel.as_posix()}'>±{w} days</a></li>")
    index_html.append("</ul></body></html>")
    (args.outdir / "index.html").write_text("\n".join(index_html), encoding="utf-8")

    print(f"\nDone. Open: {args.outdir / 'index.html'}")


if __name__ == "__main__":
    main()
