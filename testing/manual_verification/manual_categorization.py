# manual_categorization.py
#!/usr/bin/env python3
import argparse
import sys
import os
import csv
import hashlib
import datetime as dt
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

# ----------------------- Categories and subcategories -----------------------
CATS: Dict[str, List[str]] = {
    "Symptoms": [
        "Pillar symptom (inability to burp/bloating, discomfort, nausea, chest pain especially after eating, socially awkward gurgling noises, excessive flatulence, social inhibition, difficulty vomiting)",
        "Symptoms outside of these",
    ],
    "Comorbidities": [
        "Psychological (anxiety, depression, …)",
        "Physical",
    ],
    "Diagnostics": [],
    "Physicians": [],
    "Botox procedure": [
        "Dosage",
        "Repeat injection",
    ],
    "Treatments": [
        "Alternative medications",
        "Techniques",
        "Modifying diet",
    ],
    "Feelings": [
        "Emotional sensation",
        "Physical sensation",
    ],
    "Other": [],
}

BASE_COLUMNS = ["doc_index", "doc_id", "author", "created_utc", "post", "comment"]

# ----------------------- Helpers -----------------------
def fmt_time(val) -> str:
    try:
        if val is None or val == "":
            return ""
        if isinstance(val, (int, float)):
            return dt.datetime.utcfromtimestamp(val).strftime("%Y-%m-%d %H:%M UTC")
        return str(val)
    except Exception:
        return str(val)

def load_rows(parquet_path: str) -> List[Dict[str, Any]]:
    df = pd.read_parquet(parquet_path)

    if "post" not in df.columns:
        raise ValueError("Parquet must contain a 'post' column")
    if "comment" not in df.columns:
        df["comment"] = None
    if "author" not in df.columns:
        df["author"] = ""
    if "created_utc" not in df.columns:
        df["created_utc"] = ""

    mask = df["post"].astype(str).str.strip().ne("")
    df = df.loc[mask].reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for i, r in df.iterrows():
        post_text = r.get("post", "") or ""
        rows.append(
            {
                "author": r.get("author", ""),
                "created_utc": r.get("created_utc", ""),
                "post": post_text,
                "comment": r.get("comment", None),
                "doc_index": i + 1,  # 1-based
                "doc_id": hashlib.md5(post_text.encode("utf-8")).hexdigest(),
            }
        )
    return rows

def csv_columns() -> List[str]:
    cols = list(BASE_COLUMNS)
    for cat, subs in CATS.items():
        cols.append(cat)  # exact category name
        for sub in subs:
            cols.append(f"{cat}_{sub}")  # Category_Subcategory (no numbering)
    return cols

def empty_annotation_row(row: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "doc_index": row["doc_index"],
        "doc_id": row["doc_id"],
        "author": row.get("author", ""),
        "created_utc": row.get("created_utc", ""),
        "post": row.get("post", ""),
        "comment": row.get("comment", None),
    }
    for cat, subs in CATS.items():
        base[cat] = 0
        for sub in subs:
            base[f"{cat}_{sub}"] = 0
    return base

def apply_state_to_row(row_out: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    # state: {"cats": set(category names), "subs": {cat: set(sub indexes)}}
    for cat, subs in CATS.items():
        row_out[cat] = 1 if cat in state["cats"] else 0
        picks = state["subs"].get(cat, set())
        for i, sub in enumerate(subs, start=1):
            row_out[f"{cat}_{sub}"] = 1 if i in picks else 0
    return row_out

def load_or_init_csv(path: str) -> pd.DataFrame:
    cols = csv_columns()
    if os.path.exists(path):
        df = pd.read_csv(path, keep_default_na=False)
        for c in cols:
            if c not in df.columns:
                df[c] = 0 if c not in BASE_COLUMNS else ""
        df = df[cols]
        return df
    return pd.DataFrame(columns=cols)

def upsert_row(df: pd.DataFrame, row_dict: Dict[str, Any]) -> pd.DataFrame:
    idx = df.index[df["doc_id"] == row_dict["doc_id"]].tolist()
    new_row = [row_dict.get(c, None) for c in df.columns]
    if idx:
        df.loc[idx[0], :] = new_row
    else:
        df.loc[len(df)] = new_row
    return df

def any_labels_set(row: pd.Series) -> bool:
    for c in row.index:
        if c in BASE_COLUMNS:
            continue
        try:
            if int(row[c]) == 1:
                return True
        except Exception:
            pass
    return False

def find_resume_index(ann_df: pd.DataFrame, rows: List[Dict[str, Any]]) -> int:
    if ann_df is None or ann_df.empty:
        return 1
    saved = ann_df.set_index("doc_id")
    for i, r in enumerate(rows, start=1):
        did = r["doc_id"]
        if did not in saved.index:
            return i
        if not any_labels_set(saved.loc[did]):
            return i
    return 1

# Convert our in-memory doc_state to the saver format
def state_for_save(doc_state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cats": set(doc_state["cats"]),
        "subs": {k: set(v) for k, v in doc_state["subs"].items()},
    }

def save_for_index(rows: List[Dict[str, Any]], idx_1based: int, doc_state: Dict[str, Any], ann_path: str):
    if not (1 <= idx_1based <= len(rows)):
        return
    doc_prev = rows[idx_1based - 1]
    row_out = empty_annotation_row(doc_prev)
    row_out = apply_state_to_row(row_out, state_for_save(doc_state))
    df_existing = load_or_init_csv(ann_path)
    df_new = upsert_row(df_existing, row_out)
    df_new.to_csv(
        ann_path,
        index=False,
        quoting=csv.QUOTE_MINIMAL,  # commas/newlines safe
        lineterminator="\n",
        encoding="utf-8",
    )

# ----------------------- Args -----------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        default=os.path.join(os.path.dirname(__file__), "post_comment.parquet")
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "category_annotations.csv")
    )
    # Resume defaults to True; allow disabling with --no-resume
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", dest="resume", action="store_true", help="Resume from first unannotated doc")
    group.add_argument("--no-resume", dest="resume", action="store_false", help="Start from --start index")
    parser.set_defaults(resume=True)
    parser.add_argument("--start", type=int, default=1, help="1-based start index when --no-resume is used")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args

# ----------------------- App -----------------------
def main():
    st.set_page_config(page_title="Post annotator", layout="wide")
    args = parse_args()

    rows = load_rows(args.parquet)
    total = len(rows)
    if total == 0:
        st.error("No posts found in the Parquet.")
        return

    # ----- Session state -----
    if "doc_idx" not in st.session_state:
        ann_df = load_or_init_csv(args.out)
        st.session_state.doc_idx = find_resume_index(ann_df, rows) if args.resume else max(1, min(args.start, total))
    if "cat_idx" not in st.session_state:
        st.session_state.cat_idx = 0
    # NEW: per-document state store
    if "doc_states" not in st.session_state:
        st.session_state.doc_states = {}  # {doc_id: {"cats": set[str], "subs": {cat: set[int]}}}
    if "restored_doc_id" not in st.session_state:
        st.session_state.restored_doc_id = None

    ann_path = args.out

    # Navigation helpers
    def current_doc_id():
        return rows[st.session_state.doc_idx - 1]["doc_id"]

    def ensure_doc_state(doc_id: str):
        """Create/restore a doc_state for doc_id from CSV if not present."""
        if doc_id in st.session_state.doc_states:
            return st.session_state.doc_states[doc_id]

        # Initialize empty
        ds = {"cats": set(), "subs": {}}

        # If exists in CSV, hydrate
        ann_df = load_or_init_csv(ann_path)
        existing = ann_df[ann_df["doc_id"] == doc_id]
        if not existing.empty:
            rec = existing.iloc[0]
            for cat, subs in CATS.items():
                try:
                    if int(rec.get(cat, 0)) == 1:
                        ds["cats"].add(cat)
                except Exception:
                    pass
                picks = set()
                for i, sub in enumerate(subs, start=1):
                    try:
                        if int(rec.get(f"{cat}_{sub}", 0)) == 1:
                            picks.add(i)
                    except Exception:
                        pass
                if picks:
                    ds["subs"][cat] = picks

        st.session_state.doc_states[doc_id] = ds
        return ds

    def save_current_doc():
        doc_id = current_doc_id()
        ds = ensure_doc_state(doc_id)
        save_for_index(rows, st.session_state.doc_idx, ds, ann_path)

    def goto_doc(new_idx: int):
        # Save current doc to CSV, then switch and force a rerun so text updates immediately
        save_current_doc()
        st.session_state.doc_idx = max(1, min(total, new_idx))
        st.session_state.cat_idx = 0
        st.session_state.restored_doc_id = None
        st.rerun()

    def next_cat():
        st.session_state.cat_idx = min(len(CATS) - 1, st.session_state.cat_idx + 1)

    def prev_cat():
        st.session_state.cat_idx = max(0, st.session_state.cat_idx - 1)

    # ===================== LOAD CURRENT DOC & STATE =====================
    doc = rows[st.session_state.doc_idx - 1]
    doc_id = doc["doc_id"]
    ds = ensure_doc_state(doc_id)  # ds = {"cats": set(), "subs": {cat: set[int]}}

    # ----- Layout: two columns -----
    left, right = st.columns([3, 2], gap="large")

    # ===================== LEFT PANEL (post text) =====================
    with left:
        st.subheader("Post")
        body_html = (doc["post"] or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        st.markdown(
            f"""
            <div style="
                white-space: pre-wrap;
                line-height: 1.6;
                font-size: 18px;
                color: #ffffff;
                background: #000000;
                font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;
                padding: 16px;
                border: 1px solid #333333;
                border-radius: 8px;
                max-height: 80vh;
                overflow: auto;">
                {body_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ===================== RIGHT PANEL (controls + category UI) =====================
    with right:
        # Four nav buttons + Save
        bcols = st.columns([1, 1, 1, 1, 1], gap="small")
        with bcols[0]:
            if st.button("◀ Doc", use_container_width=True):
                goto_doc(st.session_state.doc_idx - 1)
        with bcols[1]:
            if st.button("Doc ▶", use_container_width=True):
                goto_doc(st.session_state.doc_idx + 1)
        with bcols[2]:
            if st.button("◀ Cat", use_container_width=True):
                prev_cat()
        with bcols[3]:
            if st.button("Cat ▶", use_container_width=True):
                next_cat()
        with bcols[4]:
            if st.button("💾 Save", use_container_width=True):
                save_current_doc()
                st.success("Saved", icon="✅")

        st.caption(
            f"Doc {st.session_state.doc_idx}/{total} · "
            f"Author: {doc.get('author','')} · {fmt_time(doc.get('created_utc'))}"
        )

        categories = list(CATS.keys())
        current_cat = categories[st.session_state.cat_idx]
        st.markdown(f"### {current_cat}")

        # ---------- Category checkbox ----------
        cat_key = f"cat::{doc_id}::{current_cat}"
        # Seed widget key only if missing; never overwrite on every run
        if cat_key not in st.session_state:
            st.session_state[cat_key] = (current_cat in ds["cats"])

        prev_present = (current_cat in ds["cats"])
        st.checkbox("Mark category present", key=cat_key)
        now_present = bool(st.session_state[cat_key])

        # Update in-memory doc_state from widget (NO CSV write here)
        if now_present and not prev_present:
            ds["cats"].add(current_cat)
        elif not now_present and prev_present:
            ds["cats"].discard(current_cat)
            # Clear sub selections for this category in memory
            ds["subs"].pop(current_cat, None)
            # Also clear any existing sub widget keys for this doc+cat
            for i in range(1, len(CATS[current_cat]) + 1):
                st.session_state[f"sub::{doc_id}::{current_cat}::{i}"] = False

        # ---------- Subcategories (persist while switching categories) ----------
        subs = CATS[current_cat]
        if (current_cat in ds["cats"]) and subs:
            st.markdown("**Subcategories**")
            picks = ds["subs"].get(current_cat, set())
            new_picks = set()
            for i, sub in enumerate(subs, start=1):
                skey = f"sub::{doc_id}::{current_cat}::{i}"
                if skey not in st.session_state:
                    st.session_state[skey] = (i in picks)
                st.checkbox(sub, key=skey)
                if st.session_state[skey]:
                    new_picks.add(i)
            ds["subs"][current_cat] = new_picks  # persist in memory

    with st.expander("CSV schema & behavior"):
        st.write(
            "- **Next/Prev Doc** automatically saves the current document before switching and reruns the app.\n"
            "- **Next/Prev Cat** switches category **without writing to CSV**; selections persist in memory for this doc.\n"
            "- Use **Save** any time to write the current doc to CSV manually.\n"
            "- Returning to a doc restores previous selections from CSV."
        )

if __name__ == "__main__":
    main()
