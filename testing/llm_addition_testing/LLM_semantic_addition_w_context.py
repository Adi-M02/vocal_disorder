#!/usr/bin/env python3
"""
Single-term evaluator with CORPUS CONTEXT (STRICT one-candidate schema)
======================================================================

What this does
--------------
- Evaluates **one expansion term at a time** against its seed using an LLM.
- Provides the LLM with:
  • DOMAIN CONTEXT (R-CPD on Reddit),
  • DEFINITIONS of relation types (hypernym/hyponym/holonym/cohyponym, synonyms, etc.),
  • INSTRUCTIONS + STRICT output schema (either [] or ["<exact candidate>"]),
  • OPTIONAL ANCHOR TERMS (global list of seed terms for background context, only seeds with non-empty expansions),
  • CORPUS CONTEXT WINDOWS for both SEED and CANDIDATE (from a local cached corpus).

Notes
-----
- There is **no closure/second pass** logic here. One pass only, per (seed, candidate) pair.
- Corpus context is fetched by dynamically importing a function (default name
  `query_from_cache_for_terms`) from a user-provided module. This function must return
  `{term: List[List[str]]}` where each inner list is a token window around the term.

Outputs (same layout as previous script family)
----------------------------------------------
  • decisions.ndjson                (per-pair audit)
  • seeds_accepted.ndjson          (one line per seed with accepted list + subset check)
  • accepted_by_seed.json          { seed: [accepted terms] }
  • filtered_expansions.json       { seed: [accepted terms] } (same as above)
  • accepted_all_flat.json         [all accepted terms across all seeds, in encounter order]
  • accepted_aligned_by_seed.json  { seed: [accepted-or-empty-string aligned to input order] }
  • summary.json                   run stats + settings
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

# -----------------------------
# Shared defaults
# -----------------------------

GLOBAL_CONTEXT_DEFAULT = (
    "The domain is R-CPD (Retrograde Cricopharyngeus Dysfunction: inability to burp) on Reddit. "
    "Expansions may include medical terminology, anatomy/physiology terms (e.g., cricopharyngeus, UES, esophagus), "
    "diagnostics (ENT visits, manometry, FEES, barium swallow, endoscopy), interventions/treatments "
    "(botox to cricopharyngeus/UES, dilation, therapy), related symptoms (chest pressure, bloating, gurgling, hiccups, "
    "nausea, reflux-like sensations), patient emotions (anxiety, embarrassment, frustration, relief, validation), "
    "actions/behaviors (massage, breathing techniques, carbonation tests, dietary changes, booking appointments), "
    "logistics (insurance, referrals, clinic names), and community/platform references (subreddit, reddit, tiktok, instagram), "
    "abbreviations of common medical terms, and reasonable umbrella/hypernym lay terms (e.g., heart_condition, throat_condition)."
)

WORDNET_STYLE_DEFINITIONS = (
    "RELATION DEFINITIONS (WordNet-style):"
    "• hypernym: the candidate is a broader umbrella of the seed (e.g., tachycardia → heart_condition)."
    "• hyponym: the candidate is a more specific instance/type of the seed."
    "• holonym: the candidate is a whole that includes the seed as a part/member."
    "• cohyponym: the candidate is a sibling under the same umbrella as the seed."
    "Also consider synonyms, near-synonyms, and morphological variants (noun/verb/adj forms of the same phenomenon)."
)

# -----------------------------
# Utils
# -----------------------------

def percentile(arr: List[int], p: float) -> float:
    if not arr:
        return 0.0
    arr = sorted(arr)
    k = (len(arr) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(arr[int(k)])
    return arr[f] * (c - k) + arr[c] * (k - f)

# -----------------------------
# LLM client (Ollama) — STRICT single-candidate protocol + corpus context
# -----------------------------

class LlmDeciderWithContext:
    def __init__(
        self,
        model: str,
        url: str = "http://localhost:11434/api/chat",
        temperature: float = 0.0,
        tokens: int = 4096,
        timeout: int = 60,
        global_context: str = GLOBAL_CONTEXT_DEFAULT,
        session: Optional[requests.Session] = None,
    ):
        self.url = url
        self.timeout = timeout
        self.global_context = (global_context or GLOBAL_CONTEXT_DEFAULT).strip()
        self.session = session or requests.Session()

        self.base_payload = {
            "model": model,
            "options": {"temperature": float(temperature), "num_ctx": int(tokens)},
            "stream": False,
        }

        # System prompt bundles domain context + definitions + strict schema rules
        self.system_prompt = (
            "You are a semantic similarity decider for short terms and underscore-separated MWEs."
            f"DOMAIN CONTEXT:{self.global_context}"
            f"{WORDNET_STYLE_DEFINITIONS}"
            "TASK (STRICT):"
            "Given a SEED and EXACTLY ONE CANDIDATE, return STRICT JSON with keys {seed, decisions}. "
            "The ONLY valid outputs for 'decisions' are either [] (reject) or [<exact candidate>] (accept)."
            "Do NOT alter, normalize, or invent strings. Return no explanations."
            "You will also receive: (a) optional ANCHOR_TERMS (a background list of domain-relevant seeds), and"
            "(b) CORPUS CONTEXT WINDOWS for both SEED and CANDIDATE — each window is a short base-token span"
            "from a Reddit corpus. Use these windows as evidence of *how* the terms are used by people."
            "Be conservative: prefer clear domain relevance over tenuous associations."
        )

    def _build_single_candidate_schema(self, candidate: str) -> dict:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["seed", "decisions"],
            "properties": {
                "seed": {"type": "string"},
                "decisions": {
                    "oneOf": [
                        {"type": "array", "maxItems": 0},
                        {"type": "array", "minItems": 1, "maxItems": 1, "items": {"const": candidate}},
                    ]
                },
            },
        }

    @staticmethod
    def _fmt_windows(windows: List[List[str]], max_windows: int, max_chars: int) -> str:
        if not windows:
            return "(none)"
        out_lines: List[str] = []
        for win in windows[: max_windows]:
            s = " ".join([str(t) for t in win])
            if len(s) > max_chars:
                s = s[: max_chars - 1] + "…"
            out_lines.append(f"- {s}")
        return " ".join(out_lines) if out_lines else "(none)"

    def _build_user_prompt(self,
                           seed: str,
                           candidate: str,
                           anchors: List[str],
                           seed_windows: List[List[str]],
                           cand_windows: List[List[str]],
                           max_anchors: int = 200,
                           max_windows: int = 12,
                           max_window_chars: int = 160) -> str:
        anc_str = " ".join(f"- {a}" for a in anchors[:max_anchors]) if anchors else "(none)"
        seed_ctx = self._fmt_windows(seed_windows, max_windows, max_window_chars)
        cand_ctx = self._fmt_windows(cand_windows, max_windows, max_window_chars)
        base = (
            f"SEED: {seed}"
            f"CANDIDATE (evaluate only this exact string; accept ⇒ return it, reject ⇒ return []):"
            f"- {candidate}"
            f"ANCHOR_TERMS (background list, optional): {anc_str}"
            f"SEED_CORPUS_WINDOWS (usage contexts):{seed_ctx}"
            f"CANDIDATE_CORPUS_WINDOWS (usage contexts):{cand_ctx}"
        )
        return base + "Respond ONLY with JSON {\"seed\": <seed>, \"decisions\": [] or [<exact candidate>]}."

    def _post(self, payload: dict) -> dict:
        resp = self.session.post(
            self.url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        if isinstance(body, dict) and "message" in body and isinstance(body["message"], dict):
            content = body["message"].get("content", "")
            return json.loads(content)
        if isinstance(body, dict) and "choices" in body:
            content = body["choices"][0]["message"]["content"]
            return json.loads(content)
        if isinstance(body, dict):
            return body
        raise ValueError("Unexpected LLM response format")

    @staticmethod
    def _coerce_terms(out: dict) -> Tuple[List[str], str]:
        raw = out.get("decisions", []) if isinstance(out, dict) else []
        if not isinstance(raw, list):
            return [], "empty"
        if raw and isinstance(raw[0], dict):
            terms: List[str] = []
            for item in raw:
                if isinstance(item, dict) and isinstance(item.get("term"), str):
                    terms.append(item["term"])
            return terms, "object->string"
        return [t for t in raw if isinstance(t, str)], "string"

    @staticmethod
    def _verify_unchanged(candidate: str, returned_terms: List[str]) -> Dict[str, List[str]]:
        cand_set = {candidate}
        decided = [t for t in returned_terms if isinstance(t, str)]
        unknown = [t for t in decided if t not in cand_set]
        seen = {}
        dupes: List[str] = []
        for t in decided:
            seen[t] = seen.get(t, 0) + 1
            if seen[t] == 2:
                dupes.append(t)
        return {"unknown_terms": sorted(set(unknown)), "duplicates": sorted(dupes)}

    def judge(self,
              seed: str,
              candidate: str,
              anchors: List[str],
              seed_windows: List[List[str]],
              cand_windows: List[List[str]]) -> Tuple[bool, Dict]:
        payload = dict(self.base_payload)
        payload["format"] = self._build_single_candidate_schema(candidate)
        payload["messages"] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_user_prompt(seed, candidate, anchors, seed_windows, cand_windows)},
        ]
        out = self._post(payload)
        terms, schema_used = self._coerce_terms(out)
        returned_seed = out.get("seed") if isinstance(out, dict) else None
        ver = self._verify_unchanged(candidate, terms)
        ver["schema_used"] = schema_used
        ver["seed_echo_ok"] = (returned_seed == seed)
        accepted = (len(terms) == 1 and terms[0] == candidate)
        schema_ok = (len(terms) in (0, 1)) and not ver["unknown_terms"] and not ver["duplicates"] and ver["seed_echo_ok"]
        ver["schema_ok"] = schema_ok
        return accepted, ver

# -----------------------------
# Data model for per-term decisions
# -----------------------------

@dataclass
class DecisionRecord:
    seed: str
    candidate: str
    accepted: bool
    decision: str  # "accept" | "reject" | "error" | "unknown_mismatch"
    schema_used: str = "unknown"
    unknown_terms: List[str] = field(default_factory=list)
    duplicates: List[str] = field(default_factory=list)
    attempts: int = 1
    latency_ms: int = 0
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

# -----------------------------
# I/O helpers
# -----------------------------

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("single_term_eval_corpusctx")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    return logger

def write_ndjson(path: Path, records: List[DecisionRecord]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")

def write_seed_ndjson(path: Path, rows: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# -----------------------------
# Runner (sequential across seeds)
# -----------------------------

class SingleTermRunner:
    def __init__(self, args: argparse.Namespace, anchors: Optional[List[str]] = None):
        self.args = args
        self.session = requests.Session()
        self.judge = LlmDeciderWithContext(
            model=args.model,
            url=args.url,
            temperature=args.temperature,
            tokens=args.tokens,
            timeout=args.timeout,
            global_context=args.global_context,
            session=self.session,
        )
        self.anchors: List[str] = list(anchors or [])

        # Load/prepare corpus context function
        if not args.context_module:
            raise ValueError("--context_module is required (module that defines query_from_cache_for_terms)")
        mod = importlib.import_module(args.context_module)
        fn_name = args.context_fn or "query_from_cache_for_terms"
        if not hasattr(mod, fn_name):
            raise AttributeError(f"Module '{args.context_module}' does not define '{fn_name}'")
        self.query_ctx_fn = getattr(mod, fn_name)

        # Preload windows for ALL terms to avoid re-loading cache repeatedly
        expansions = load_json(args.expansions)
        all_terms: Set[str] = set()
        for seed, cand_list in expansions.items():
            if not isinstance(cand_list, list):
                continue
            cands = [str(c) for c in cand_list]
            if args.limit_per_seed and args.limit_per_seed > 0:
                cands = cands[: args.limit_per_seed]
            all_terms.add(str(seed))
            all_terms.update(cands)
        self.term_windows: Dict[str, List[List[str]]] = self.query_ctx_fn(
            cache_path=args.cache_path,
            terms=sorted(all_terms),
            k=args.k,
            window=args.window,
            format=args.context_format,
            setup_module=(args.context_setup_module or ""),
        )

    def _windows_for(self, term: str) -> List[List[str]]:
        wins = self.term_windows.get(term)
        return wins if isinstance(wins, list) else []

    def _call_with_retries(self, fn, *args, **kwargs) -> Tuple[bool, Dict, int, Optional[str]]:
        max_tries = max(1, int(self.args.retries) + 1)
        delay = 0.6
        for attempt in range(1, max_tries + 1):
            t0 = time.time()
            try:
                ok, ver = fn(*args, **kwargs)
                ms = int((time.time() - t0) * 1000)
                return ok, ver, ms, None
            except Exception as e:
                ms = int((time.time() - t0) * 1000)
                if attempt >= max_tries:
                    return False, {"schema_used": "?", "unknown_terms": [], "duplicates": [], "seed_echo_ok": False}, ms, str(e)
                time.sleep(min(delay * (2 ** (attempt - 1)) * (1.0 + 0.25 * random.random()), 5.0))
        return False, {"schema_used": "?", "unknown_terms": [], "duplicates": [], "seed_echo_ok": False}, 0, "unknown"

    def eval_pair(self, seed: str, candidate: str) -> DecisionRecord:
        if not isinstance(seed, str) or not isinstance(candidate, str) or not seed or not candidate:
            return DecisionRecord(seed=str(seed), candidate=str(candidate), accepted=False, decision="error", error="invalid seed/candidate")
        seed_w = self._windows_for(seed)
        cand_w = self._windows_for(candidate)
        ok, ver, ms, err = self._call_with_retries(self.judge.judge, seed, candidate, self.anchors, seed_w, cand_w)
        if err is not None:
            return DecisionRecord(
                seed=seed,
                candidate=candidate,
                accepted=False,
                decision="error",
                schema_used=ver.get("schema_used", "unknown"),
                unknown_terms=ver.get("unknown_terms", []),
                duplicates=ver.get("duplicates", []),
                attempts=max(1, int(self.args.retries) + 1),
                latency_ms=ms,
                error=str(err),
            )
        schema_ok = ver.get("schema_ok", True)
        decision = "accept" if ok else ("unknown_mismatch" if (not schema_ok or ver.get("unknown_terms") or ver.get("duplicates")) else "reject")
        return DecisionRecord(
            seed=seed,
            candidate=candidate,
            accepted=ok,
            decision=decision,
            schema_used=ver.get("schema_used", "unknown"),
            unknown_terms=ver.get("unknown_terms", []),
            duplicates=ver.get("duplicates", []),
            attempts=1,
            latency_ms=ms,
            error=None,
        )

# -----------------------------
# Logging summary
# -----------------------------

def log_summary(logger: logging.Logger, all_records: List[DecisionRecord], accepted_by_seed: Dict[str, List[str]]):
    total = len(all_records)
    acc = sum(1 for r in all_records if r.decision == "accept")
    rej = sum(1 for r in all_records if r.decision == "reject")
    unk = sum(1 for r in all_records if r.decision == "unknown_mismatch")
    err = sum(1 for r in all_records if r.decision == "error")

    logger.info("# Single-term LLM evaluation summary (with corpus context)")
    logger.info(f"Total evals: {total} | accept={acc} reject={rej} unknown_mismatch={unk} error={err}")
    lat_ok = [r.latency_ms for r in all_records if r.decision in {"accept", "reject", "unknown_mismatch"}]
    if lat_ok:
        logger.info(
            f"Latency (ms) median={int(percentile(lat_ok,50))} p90={int(percentile(lat_ok,90))} max={max(lat_ok)}"
        )
    logger.info("")

    logger.info("## per-seed acceptance rates")
    for seed in sorted(accepted_by_seed.keys()):
        seed_recs = [r for r in all_records if r.seed == seed]
        if not seed_recs:
            continue
        acc_s = sum(1 for r in seed_recs if r.decision == "accept")
        rej_s = sum(1 for r in seed_recs if r.decision == "reject")
        unk_s = sum(1 for r in seed_recs if r.decision == "unknown_mismatch")
        err_s = sum(1 for r in seed_recs if r.decision == "error")
        logger.info(
            f"{seed:<28s} | total={len(seed_recs):4d} accept={acc_s:4d} reject={rej_s:4d} unknown={unk_s:3d} error={err_s:3d}"
        )

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Single-term LLM judgments with corpus context (STRICT schema, R-CPD aware).")
    ap.add_argument("--expansions", required=True, help="Path to expansions JSON {seed: [terms]}")
    ap.add_argument("--outdir", required=True, help="Directory for outputs")

    # LLM
    ap.add_argument("--model", type=str, default="llama3.3:latest", help="Ollama model name")
    ap.add_argument("--url", type=str, default="http://localhost:11436/api/chat", help="Ollama chat endpoint")
    ap.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    ap.add_argument("--tokens", type=int, default=8196, help="LLM context tokens")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    ap.add_argument("--global_context", type=str, default=GLOBAL_CONTEXT_DEFAULT, help="Domain context injected into the system prompt")

    # Execution controls
    ap.add_argument("--retries", type=int, default=2, help="Retry count on errors (exponential backoff)")
    ap.add_argument("--sort", action="store_true", help="Sort final accepted lists alphabetically where applicable")
    ap.add_argument("--limit_per_seed", type=int, default=0, help="Limit number of candidates per seed (0=all)")

    # Anchors (optional background list): if set, use all expansion keys that have a non-empty list
    ap.add_argument("--use_anchors", action="store_true", help="Include anchors = expansions' keys that have non-empty lists")

    # Corpus context module + cache settings
    ap.add_argument("--context_module", type=str, default='testing/adding_context/context_api.py', help="Module path that defines query_from_cache_for_terms")
    ap.add_argument("--context_fn", type=str, default="query_from_cache_for_terms", help="Function name to call in the context module")
    ap.add_argument("--cache_path", type=str, required=True, help="Path to cached corpus (e.g., parquet)")
    ap.add_argument("--context_format", type=str, default="parquet", help="Cache format (e.g., parquet)")
    ap.add_argument("--context_setup_module", type=str, default="", help="Optional setup module name consumed by the context function")
    ap.add_argument("--k", type=int, default=8, help="Number of windows per term to include")
    ap.add_argument("--window", type=int, default=6, help="Half-window size (tokens to left/right) or implementation-defined")

    args = ap.parse_args()

    expansions = load_json(args.expansions)
    if not isinstance(expansions, dict):
        raise ValueError("Expansions JSON must be an object mapping seeds to lists of terms.")

    # Prepare global anchors (full seed vocabulary) if requested — only seeds with non-empty lists
    anchors: List[str] = sorted([str(s) for s, v in expansions.items() if isinstance(v, list) and v]) if args.use_anchors else []

    # Prepare output dir
    outdir = Path(args.outdir).expanduser().resolve()
    timestamp = datetime.now().strftime("%m_%d_%H_%M")
    eval_dir = outdir / f"single_term_eval_corpusctx_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_path = eval_dir / "single_term_eval.log"
    ndjson_pairs_path = eval_dir / "decisions.ndjson"
    ndjson_seeds_path = eval_dir / "seeds_accepted.ndjson"
    accepted_by_seed_path = eval_dir / "accepted_by_seed.json"
    filtered_expansions_path = eval_dir / "filtered_expansions.json"
    accepted_all_flat_path = eval_dir / "accepted_all_flat.json"
    accepted_aligned_path = eval_dir / "accepted_aligned_by_seed.json"
    summary_path = eval_dir / "summary.json"

    logger = setup_logger(log_path)
    runner = SingleTermRunner(args, anchors=anchors)

    # Build jobs
    seeds = list(expansions.keys())
    jobs: List[Tuple[str, List[str]]] = []
    for seed in seeds:
        cands = expansions.get(seed, [])
        if not isinstance(cands, list):
            logger.info(f"[warn] Seed '{seed}' expansions not a list; treating as empty.")
            cands = []
        else:
            cands = [str(c) for c in cands]
        if args.limit_per_seed and args.limit_per_seed > 0:
            cands = cands[: args.limit_per_seed]
        jobs.append((seed, cands))

    # Process sequentially
    all_records: List[DecisionRecord] = []
    accepted_by_seed: Dict[str, List[str]] = {}
    aligned_by_seed: Dict[str, List[str]] = {}
    per_seed_rows: List[dict] = []

    logger.info("# Processing seeds sequentially (one LLM call per candidate)")
    for seed, cand_list in jobs:
        acc_set: Set[str] = set()
        aligned: List[str] = []
        recs: List[DecisionRecord] = []

        for cand in cand_list:
            rec = runner.eval_pair(seed, cand)
            recs.append(rec)
            if rec.accepted:
                acc_set.add(cand)
                aligned.append(cand)
            else:
                aligned.append("")

        # Subset check
        original_set = set(expansions.get(seed, []) if isinstance(expansions.get(seed), list) else [])
        violations = sorted([t for t in acc_set if t not in original_set])
        if violations:
            acc_set = {t for t in acc_set if t in original_set}
        acc_list = sorted(acc_set) if args.sort else list(acc_set)

        accepted_by_seed[seed] = acc_list
        aligned_by_seed[seed] = aligned
        per_seed_rows.append({
            "seed": seed,
            "accepted": acc_list,
            "checked_subset": len(violations) == 0,
            "violations": violations,
        })
        all_records.extend(recs)

    # Write outputs
    write_ndjson(ndjson_pairs_path, all_records)
    write_seed_ndjson(ndjson_seeds_path, per_seed_rows)
    save_json(accepted_by_seed_path, accepted_by_seed)
    save_json(filtered_expansions_path, accepted_by_seed)
    save_json(accepted_aligned_path, aligned_by_seed)

    accepted_all_flat: List[str] = []
    for seed, _cands in jobs:
        aligned = aligned_by_seed.get(seed, [])
        for t in aligned:
            if t:
                accepted_all_flat.append(t)
    save_json(accepted_all_flat_path, accepted_all_flat)

    # Summary
    seeds_with_errors = len({r.seed for r in all_records if r.decision == "error"})
    seeds_all_empty = sum(1 for s in seeds if not accepted_by_seed.get(s))
    summary = {
        "total_evals": len(all_records),
        "accept": sum(1 for r in all_records if r.decision == "accept"),
        "reject": sum(1 for r in all_records if r.decision == "reject"),
        "unknown_mismatch": sum(1 for r in all_records if r.decision == "unknown_mismatch"),
        "error": sum(1 for r in all_records if r.decision == "error"),
        "seeds_with_errors": seeds_with_errors,
        "seeds_all_empty": seeds_all_empty,
        "latency_ms_median": int(percentile([r.latency_ms for r in all_records if r.latency_ms], 50)),
        "latency_ms_p90": int(percentile([r.latency_ms for r in all_records if r.latency_ms], 90)),
        "out_dir": str(eval_dir.resolve()),
        "model": args.model,
        "url": args.url,
        "temperature": args.temperature,
        "tokens": args.tokens,
        "timeout": args.timeout,
        "retries": args.retries,
        "sorted": bool(args.sort),
        "use_anchors": bool(args.use_anchors),
        "anchors_count": len(anchors),
        "context_module": args.context_module,
        "context_fn": args.context_fn,
        "cache_path": args.cache_path,
        "k": args.k,
        "window": args.window,
    }
    save_json(summary_path, summary)

    # Human-readable summary
    logger.info(f"# Output dir: {eval_dir.resolve()}")
    logger.info(f"# Pair decisions : {ndjson_pairs_path.resolve()}")
    logger.info(f"# Seed rows      : {ndjson_seeds_path.resolve()}")
    logger.info(f"# Accepted map   : {accepted_by_seed_path.resolve()}")
    logger.info(f"# Filtered exp   : {filtered_expansions_path.resolve()}")
    logger.info(f"# Accepted flat  : {accepted_all_flat_path.resolve()}")
    logger.info(f"# Accepted align : {accepted_aligned_path.resolve()}")
    logger.info(f"# Summary        : {summary_path.resolve()}")
    logger.info("")
    log_summary(logger, all_records, accepted_by_seed)

if __name__ == "__main__":
    main()
