# streamlit_parquet_viewer.py
#!/usr/bin/env python3
import argparse
import sys
from typing import List, Dict, Any
import datetime as dt

import pandas as pd
import streamlit as st


def fmt_created_utc(val) -> str:
    try:
        if val is None or val == "":
            return ""
        if isinstance(val, (int, float)):
            return dt.datetime.utcfromtimestamp(val).strftime("%Y-%m-%d %H:%M UTC")
        return str(val)
    except Exception:
        return str(val)


def coalesce_text(row: Dict[str, Any]) -> str:
    p = row.get("post")
    c = row.get("comment")
    if isinstance(p, str) and p.strip():
        return p
    if isinstance(c, str) and c.strip():
        return c
    return ""


def load_rows(parquet_path: str) -> List[Dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    expected = ["author", "created_utc", "post", "comment"]
    rows = []
    for _, r in df.iterrows():
        d = {c: r.get(c, None) for c in expected if c in df.columns}
        d.setdefault("author", "")
        d.setdefault("created_utc", "")
        d.setdefault("post", None)
        d.setdefault("comment", None)
        d["text"] = coalesce_text(d)
        d["kind"] = "post" if (isinstance(d.get("post"), str) and d["post"]) else "comment"
        rows.append(d)
    return rows


def parse_args():
    # Streamlit passes args after a "--"
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default='post_comment.parquet')
    parser.add_argument("--start", type=int, default=0)
    # Only parse known so Streamlit’s own args are ignored
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def apply_filters(rows: List[Dict[str, Any]], kind, author, query):
    data = rows
    if kind != "all":
        data = [r for r in data if r["kind"] == kind]
    if author:
        a = author.strip().lower()
        data = [r for r in data if (r.get("author") or "").lower().find(a) != -1]
    if query:
        q = query.strip().lower()
        data = [r for r in data if r["text"].lower().find(q) != -1]
    return data


def main():
    st.set_page_config(page_title="noburp Parquet Viewer", layout="wide")
    args = parse_args()
    rows = load_rows(args.parquet)

    # session state
    if "idx" not in st.session_state:
        st.session_state.idx = max(1, min(args.start, len(rows)))
    if "kind" not in st.session_state:
        st.session_state.kind = "all"
    if "author" not in st.session_state:
        st.session_state.author = ""
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "font_px" not in st.session_state:
        st.session_state.font_px = 18

    # Sidebar filters
    st.sidebar.header("Filters")
    st.session_state.kind = st.sidebar.selectbox("Type", ["all", "post", "comment"], index=0)
    st.session_state.author = st.sidebar.text_input("Author contains", st.session_state.author)
    st.session_state.query = st.sidebar.text_input("Text contains", st.session_state.query)
    st.session_state.font_px = st.sidebar.slider("Font size", 12, 36, st.session_state.font_px)

    filtered = apply_filters(rows, st.session_state.kind, st.session_state.author, st.session_state.query)
    total = len(filtered)

    st.title("noburp Parquet Viewer")
    st.caption(f"Loaded {len(rows)} rows from `{args.parquet}` - showing {total} after filters")

    if total == 0:
        st.info("No documents match the filters.")
        return

    # Clamp index
    st.session_state.idx = max(1, min(st.session_state.idx, total))

    # Controls
    c1, c2, c3, c4, c5 = st.columns([1, 1, 3, 1, 1])
    with c1:
        if st.button("⏮ First"):
            st.session_state.idx = 1
    with c2:
        if st.button("◀ Prev"):
            st.session_state.idx = max(1, st.session_state.idx - 1)
    with c3:
        st.session_state.idx = st.slider("Document index", 1, total, st.session_state.idx, key="slider_idx")
    with c4:
        if st.button("Next ▶"):
            st.session_state.idx = min(total, st.session_state.idx + 1)
    with c5:
        if st.button("Last ⏭"):
            st.session_state.idx = total

    row = filtered[st.session_state.idx - 1]
    meta_left = f"**Author:** {row.get('author','')}"
    meta_mid = f"**When:** {fmt_created_utc(row.get('created_utc'))}"
    meta_right = f"**Doc:** {st.session_state.idx}/{total}  **Type:** {row.get('kind','')}"
    st.markdown(
        f"""
        <div style="padding:8px;border-bottom:2px solid #4b9cd3;background:#f6f6f6;">
          <span style="margin-right:24px;">{meta_left}</span>
          <span style="margin-right:24px;">{meta_mid}</span>
          <span>{meta_right}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <style>
        .docbox {{
            white-space: pre-wrap;
            line-height: 1.5;
            font-size: {st.session_state.font_px}px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
            padding: 16px;
            background: #fcfcfc;
            border: 1px solid #e5e5e5;
            border-radius: 8px;
            max-height: 70vh;
            overflow: auto;
        }}
        </style>
        <div class="docbox">{row["text"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")}</div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Help"):
        st.write(
            "Use the buttons or slider to move between documents. "
            "Use the sidebar to filter and change font size."
        )


if __name__ == "__main__":
    main()
