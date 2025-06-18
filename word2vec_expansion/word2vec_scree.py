#!/usr/bin/env python3
"""
Train CBOW & Skip-gram Word2Vec models at dims 100–500, then
plot MDS stress vs. # of MDS dimensions (1…mds_max_dim) using Plotly,
evaluated on your rcpd_terms_6_5.json terms (spell-checked & lemmatized).
"""
import sys, os, json, time, datetime, argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from gensim.models import Word2Vec

# assume your project root has ../vocal_disorder on PYTHONPATH
sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize
try:
    from spellchecker_folder.spellchecker import spellcheck_token_list
except ImportError:
    spellcheck_token_list = None

def load_terms(path: str) -> list[str]:
    """Flatten all values in JSON dict to a single list of terms."""
    with open(path, 'r', encoding='utf-8') as f:
        term_map = json.load(f)
    terms = []
    for lst in term_map.values():
        terms.extend(lst)
    return terms

def load_lookup(path: str) -> dict[str,str]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    p = argparse.ArgumentParser(description="Word2Vec + Plotly MDS Stress")
    p.add_argument("--spellcheck", action="store_true",
                   help="Use spell-checked tokenizer")
    p.add_argument("--lookup", type=str, default="testing/lemma_lookup.json",
                   help="Path to lemma lookup JSON")
    p.add_argument("--vector_sizes", type=str, default="100,200,300,400,500",
                   help="Comma-separated embedding dims")
    p.add_argument("--window", type=int, default=7, help="Context window")
    p.add_argument("--min_count", type=int, default=5, help="Min token freq")
    p.add_argument("--epochs", type=int, default=5, help="Training epochs")
    p.add_argument("--mds_max_dim", type=int, default=10,
                   help="Max # of MDS dims to evaluate")
    args = p.parse_args()

    sizes = [int(s) for s in args.vector_sizes.split(",")]
    lookup_map = load_lookup(args.lookup)
    raw_terms = load_terms("rcpd_terms_6_5.json")

    # choose tokenizer
    if args.spellcheck:
        if spellcheck_token_list is None:
            raise RuntimeError("Spell-checker module not found")
        print("→ Using spell-check tokenizer")
        def token_fn(txt):
            return spellcheck_token_list(clean_and_tokenize(txt))
    else:
        print("→ Using vanilla tokenizer")
        token_fn = clean_and_tokenize

    # fetch & clean Reddit docs
    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        mongo_uri="mongodb://localhost:27017/",
    )
    print(f"Fetched {len(docs)} Reddit documents")
    cleaned = []
    for t in docs:
        toks = token_fn(t)
        cleaned.append([ lookup_map.get(tok, tok) for tok in toks ])
    print(f"Tokenized & lemmatized {len(cleaned)} docs")

    # prepare output folder
    out_dir = Path("scree")
    out_dir.mkdir(exist_ok=True)

    # save run info
    info = {**vars(args), "timestamp": datetime.datetime.now().isoformat()}
    with open(out_dir/"info.json", "w") as f:
        json.dump(info, f, indent=2)

    # store stress curves per arch
    for sg_flag, arch_name in [(0, "cbow"), (1, "skipgram")]:
        stress_results = {}
        for dim in sizes:
            print(f"\n=== Training {arch_name.upper()} (dim={dim}) ===")
            # train model
            start = time.time()
            model = Word2Vec(
                vector_size=dim,
                window=args.window,
                min_count=args.min_count,
                sg=sg_flag,
                workers=max(1, os.cpu_count()-1)
            )
            model.build_vocab(cleaned)
            model.train(cleaned,
                        total_examples=len(cleaned),
                        epochs=args.epochs)
            mdl_path = out_dir / f"model_{arch_name}_{dim}d.model"
            model.save(str(mdl_path))
            print(f"  • Saved model → {mdl_path}  ({time.time()-start:.1f}s)")

            # build vectors for each RC-PD term (average token vectors)
            term_vecs = []
            for term in raw_terms:
                toks = token_fn(term)
                toks = [ lookup_map.get(t, t) for t in toks ]
                vecs = [model.wv[t] for t in toks if t in model.wv]
                if len(vecs)==0:
                    continue
                term_vecs.append(np.mean(vecs, axis=0))
            n_terms = len(term_vecs)
            print(f"  • Using {n_terms} terms for MDS stress")
            if n_terms < 2:
                raise RuntimeError("Need ≥2 term vectors for MDS")

            # pairwise distances
            D = pairwise_distances(np.vstack(term_vecs), metric="euclidean")

            # compute stress for k=1…mds_max_dim
            stresses = []
            ks = list(range(1, args.mds_max_dim+1))
            for k in ks:
                mds = MDS(n_components=k,
                          dissimilarity="precomputed",
                          random_state=0,
                          n_init=1,
                          max_iter=300)
                mds.fit(D)
                stresses.append(mds.stress_)
                print(f"    k={k} → stress={mds.stress_:.0f}")
            stress_results[dim] = stresses

        # Plotly: one trace per embedding dim
        fig = go.Figure()
        for dim, stresses in stress_results.items():
            fig.add_trace(go.Scatter(
                x=ks,
                y=stresses,
                mode="lines+markers",
                name=f"{dim}d"
            ))
        fig.update_layout(
            title=f"MDS Stress vs. # Dimensions ({arch_name.upper()})",
            xaxis_title="MDS dimensions (k)",
            yaxis_title="Stress",
            legend_title="Embedding size"
        )
        html_path = out_dir / f"mds_stress_{arch_name}.html"
        fig.write_html(str(html_path))
        print(f"\n→ Plotly HTML saved to {html_path}")

    print("\nDone. Note: 108 terms is generally sufficient for an MDS stress curve—stress estimates with ~100 points remain meaningful, though more items can improve stability.")

if __name__ == "__main__":
    main()
