import argparse, json, os, sys, re
from pathlib import Path
from datetime import datetime
from gensim.models import Word2Vec

sys.path.append("../vocal_disorder")
from utils.load_json import load_json

def get_arch(model_path: Path, model: Word2Vec):
    """
    Return 'cbow' or 'skipgram'.
    1. Try to read it from filename   “…_cbow.model” / “…_skipgram.model”.
    2. Fallback: use gensim attribute  model.sg  (0 = CBOW, 1 = Skip-gram).
    """
    m = re.search(r'_(cbow|skipgram)\.model$', model_path.name)
    if m:
        return m.group(1)
    return "skipgram" if model.sg == 1 else "cbow"

def most_similar(model: Word2Vec, term: str, k: int, min_cos: float):
    """
    Return top-k neighbours (cosine) for one term.  Empty list if OOV.
    if min_cos is set, filter results to only include those with cosine >= min_cos."""
    if term not in model.wv:
        return []
    sims = model.wv.most_similar(positive=[term], topn=k)
    if min_cos is None:
        return [t for t, _ in sims]
    return [t for t, cos in sims if cos > min_cos]

def build_output_path(model_dir: Path, arch: str, k: int) -> Path:
    ts   = datetime.now().strftime("%m_%d_%H_%M")
    outd = model_dir / f"{ts}_expansion{arch}"   # ← include arch
    outd.mkdir(parents=True, exist_ok=True)
    if args.min_cos is None:
        return outd / f"topk_{k}.json"
    return outd / f"topk_{k}_min_cos_{args.min_cos}.json"

def main(args):
    model_file = Path(args.model).expanduser()
    model = Word2Vec.load(str(model_file))

    arch       = get_arch(model_file, model)

    seeds_obj  = load_json(args.seed_json)
    seed_terms = ( [t for lst in seeds_obj.values() for t in lst]
                   if isinstance(seeds_obj, dict)
                   else seeds_obj )

    results = {s: most_similar(model, s, args.topk, args.min_cos) for s in seed_terms}
    out_dir  = model_file.parent 
    out_path = build_output_path(out_dir, arch, args.topk)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote expansions for {len(results)} seeds ➜  {out_path}")

# ────────────────────────── CLI ────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Path to Word2Vec .model file (not just directory)")
    p.add_argument("--seed_json",  required=True,
                   help="JSON with seed terms (list or {cat: [terms]})")
    p.add_argument("--topk", type=int, default=25)
    p.add_argument("--min_cos", type=float, default=0.6)
    args = p.parse_args()
    main(args)
