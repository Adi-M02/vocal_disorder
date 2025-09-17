#!/usr/bin/env python3
"""
Single-term evaluator for seed-term expansions (STRICT schema) + CORPUS CONTEXT
===============================================================================

What this does
--------------
- Evaluates **one expansion term at a time** against its seed using an LLM.
- Provides ONLY: instructions, domain context (R-CPD), WordNet-style relation info,
  and corpus context passages (up to 3 longest base-text snippets for the SEED and the CANDIDATE).
- PRESERVES STRICT I/O SCHEMA: only [] (reject) or ["<exact candidate>"] (accept).

Outputs
-------
  • decisions.ndjson                (per-pair audit)
  • seeds_accepted.ndjson          (one line per seed with accepted list + subset check)
  • accepted_by_seed.json          { seed: [accepted terms] }
  • filtered_expansions.json       { seed: [accepted terms] } (same as above; mirrors input shape)
  • accepted_all_flat.json         [all accepted terms across all seeds, in encounter order]
  • accepted_aligned_by_seed.json  { seed: [accepted-or-empty-string aligned to input order] }
  • summary.json                   run stats + settings
  • log file
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple
import sys
import requests
import pandas as pd
sys.path.append("../vocal_disorder")
# Use your corpus helpers directly from your module
# (builds the base+ngram DF; returns longest base-text contexts for a term)
from testing.adding_context.get_corpus_similar import build_ngram_df, sample_docs_containing 

# Optional JSON loader from your project
try:
    from utils.io import load_json  # type: ignore
except Exception:
    def load_json(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

# -----------------------------
# Shared defaults (domain & prompt context)
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

# -----------------------------
# Small helpers
# -----------------------------

def _truncate(s: str, max_chars: int) -> str:
    s = s.strip()
    return (s[:max_chars] + "…") if len(s) > max_chars else s

def _format_context_block(seed: str, seed_ctx: List[str], cand: str, cand_ctx: List[str], max_chars: int = 320) -> str:
    seed_lines = "\n".join(f"- { _truncate(x, max_chars) }" for x in (seed_ctx or [])) or "- (no exact corpus match)"
    cand_lines = "\n".join(f"- { _truncate(x, max_chars) }" for x in (cand_ctx or [])) or "- (no exact corpus match)"
    return (
        "CONTEXT FROM CORPUS (base-text snippets; matched via n-gram index):\n"
        f"SEED '{seed}' examples (up to 3 longest):\n{seed_lines}\n\n"
        f"CANDIDATE '{cand}' examples (up to 3 longest):\n{cand_lines}\n\n"
        "Use the *semantic usage* of these terms to inform your decision. "
        "Do not invent new strings. Decide strictly per the schema."
    )

def verify_unchanged(candidates: List[str], returned_terms: List[str]) -> Dict[str, List[str]]:
    """Compare LLM-returned terms to the original candidates (exact string match)."""
    cand_set = set(candidates)
    decided_terms = [t for t in returned_terms if isinstance(t, str)]
    unknown = [t for t in decided_terms if t not in cand_set]
    seen: Dict[str, int] = {}
    dupes: List[str] = []
    for t in decided_terms:
        seen[t] = seen.get(t, 0) + 1
        if seen[t] == 2:
            dupes.append(t)
    return {"unknown_terms": sorted(set(unknown)), "duplicates": sorted(dupes)}

# -----------------------------
# Relation categories (WordNet-style)
# -----------------------------

RELATION_ENUM = [
    "hypernym",            # candidate is a broader umbrella of seed (is-a up)
    "hyponym",             # candidate is a more specific instance/type of seed (is-a down)
    "holonym",             # candidate is a whole that includes the seed as a part/member (has-a up)
    "cohyponym",           # candidate is a sibling under the same hypernym as seed
    "synonym",             # same synset or equivalent naming
    "morphological_variant",
    "near_synonym",
    "unrelated",
    "unknown",
]

# -----------------------------
# LLM client (Ollama) — STRICT single-candidate protocol
# -----------------------------

class LlmSimilarityDecider:
    """
    Single prompt: instructions + domain context + WordNet-style relations + optional corpus context.
    Strict JSON schema: {seed, decisions[, relation]} where decisions ∈ {[], ["<exact candidate>"]}.
    """
    def __init__(
        self,
        model: str,
        url: str = "http://localhost:11434/api/chat",
        temperature: float = 0.0,
        tokens: int = 2048,
        timeout: int = 60,
        include_relation: bool = False,
        global_context: str = GLOBAL_CONTEXT_DEFAULT,
        session: Optional[requests.Session] = None,
    ):
        self.url = url
        self.timeout = timeout
        self.global_context = (global_context or GLOBAL_CONTEXT_DEFAULT).strip()
        self.include_relation = bool(include_relation)
        self.session = session or requests.Session()

        self.base_payload = {
            "model": model,
            "options": {"temperature": float(temperature), "num_ctx": int(tokens)},
            "stream": False,
        }

        # Single system prompt (no anchors/closure mode)
        relation_info = (
            "Consider WordNet-style relations:\n"
            "• Synonyms / same synset / morphological variants / near-synonyms.\n"
            "• Hypernym/hyponym (is-a up/down).\n"
            "• Holonym (has-a up: whole that includes the seed as part/member).\n"
            "• Cohyponym (sibling under the same hypernym as the seed).\n"
        )
        acceptance_rules = (
            "ACCEPT if the candidate helps a user find/name/describe the same concept as the seed OR is a closely "
            "neighboring concept in this R-CPD domain, including the relations above. "
            "REJECT ultra-generic catch-alls and off-domain terms."
        )

        self.system_prompt = (
            "You are a semantic similarity decider for short terms and underscore-separated MWEs.\n"
            f"DOMAIN CONTEXT:\n{self.global_context}\n\n"
            f"{relation_info}\n"
            "TASK (STRICT):\n"
            "Given a SEED and EXACTLY ONE CANDIDATE, return STRICT JSON with keys {seed, decisions"
            + (", relation" if self.include_relation else "")
            + "}.\n"
            "The ONLY valid outputs for 'decisions' are either [] (reject) or [<exact candidate>] (accept).\n"
            "Do NOT alter, normalize, or invent strings. Return no explanations.\n\n"
            f"{acceptance_rules}\n"
        )

    def _build_single_candidate_schema(self, candidate: str) -> dict:
        props = {
            "seed": {"type": "string"},
            "decisions": {
                "oneOf": [
                    {"type": "array", "maxItems": 0},  # reject → []
                    {  # accept → [candidate]
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 1,
                        "items": {"const": candidate},
                    },
                ]
            },
        }
        required = ["seed", "decisions"]
        if self.include_relation:
            props["relation"] = {"type": "string", "enum": RELATION_ENUM}
        return {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": props,
        }

    def _build_user_prompt(self, seed: str, candidate: str, ctx_block: Optional[str]) -> str:
        base = (
            f"SEED: {seed}\n"
            f"CANDIDATE (evaluate only this exact string; accept ⇒ return it, reject ⇒ return []):\n- {candidate}\n\n"
        )
        if ctx_block:
            base += ctx_block + "\n"
        if self.include_relation:
            return base + 'Respond ONLY with JSON {"seed": <seed>, "decisions": [] or ["<exact candidate>"], optionally "relation": <one label>}.'
        else:
            return base + 'Respond ONLY with JSON {"seed": <seed>, "decisions": [] or ["<exact candidate>"]}.'

    def _post(self, payload: dict) -> dict:
        resp = self.session.post(self.url, headers={"Content-Type": "application/json"}, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        body = resp.json()
        # Common Ollama envelopes
        if isinstance(body, dict) and "message" in body and isinstance(body["message"], dict):
            content = body["message"].get("content", "")
            try:
                return json.loads(content)
            except Exception:
                pass
        if isinstance(body, dict) and "choices" in body:
            content = body["choices"][0]["message"]["content"]
            return json.loads(content)
        if isinstance(body, dict):
            return body
        raise ValueError("Unexpected Ollama response format")

    def _coerce_terms(self, out: dict) -> Tuple[List[str], str]:
        raw = out.get("decisions", []) if isinstance(out, dict) else []
        if not isinstance(raw, list):
            return [], "empty"
        if raw and isinstance(raw[0], dict):
            terms: List[str] = []
            for item in raw:
                t = item.get("term") if isinstance(item, dict) else None
                if isinstance(t, str):
                    terms.append(t)
            return terms, "object->string"
        else:
            return [t for t in raw if isinstance(t, str)], "string"

    def _extract_relation(self, out: dict) -> Optional[str]:
        rel = out.get("relation")
        return rel if isinstance(rel, str) else None

    def judge_single(self, seed: str, candidate: str, ctx_block: Optional[str]) -> Tuple[bool, Dict]:
        payload = dict(self.base_payload)
        payload["format"] = self._build_single_candidate_schema(candidate)
        payload["messages"] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_user_prompt(seed, candidate, ctx_block)},
        ]
        out = self._post(payload)
        terms, schema_used = self._coerce_terms(out)
        relation = self._extract_relation(out)
        returned_seed = out.get("seed") if isinstance(out, dict) else None
        ver = verify_unchanged([candidate], terms)
        ver["schema_used"] = schema_used
        ver["seed_echo_ok"] = (returned_seed == seed)
        ver["relation"] = relation
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
    prompt_type: str  # always "single"
    schema_used: str = "unknown"
    unknown_terms: List[str] = field(default_factory=list)
    duplicates: List[str] = field(default_factory=list)
    attempts: int = 1
    latency_ms: int = 0
    error: Optional[str] = None
    relation: Optional[str] = None  # optional when relation_mode on

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

# -----------------------------
# Runner (sequential across seeds; no concurrency)
# -----------------------------

class SingleTermRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        df: Optional[pd.DataFrame] = None,
        sample_docs_fn: Optional[Callable[..., List[str]]] = None,
    ):
        self.args = args
        self.session = requests.Session()
        self.judge = LlmSimilarityDecider(
            model=args.model,
            url=args.url,
            temperature=args.temperature,
            tokens=args.tokens,
            timeout=args.timeout,
            include_relation=bool(args.relation_mode),
            global_context=args.global_context,
            session=self.session,
        )
        self.cache: Dict[Tuple[str, str], DecisionRecord] = {}
        self.df = df
        self.sample_docs_fn = sample_docs_fn

        if args.cache_path and Path(args.cache_path).exists():
            self._load_cache(args.cache_path)

    def _load_cache(self, path: str):
        try:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = (rec["seed"], rec["candidate"])
                self.cache[key] = DecisionRecord(**rec)
        except Exception as e:
            print(f"[warn] Failed to load cache {path}: {e}")

    def _append_cache(self, path: str, rec: DecisionRecord):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(rec.to_json() + "\n")
        except Exception as e:
            print(f"[warn] Failed to write cache {path}: {e}")

    def _safe_sample(self, term: str, k: int) -> List[str]:
        """Call the provided sample_docs_containing with flexible signature."""
        if not self.sample_docs_fn or self.df is None:
            return []
        try:
            return self.sample_docs_fn(self.df, term, k, regex_text_fallback=True)
        except TypeError:
            return self.sample_docs_fn(self.df, term, k)

    def _build_ctx_block(self, seed: str, candidate: str) -> str:
        k = max(1, int(self.args.context_k))
        seed_ctx = self._safe_sample(seed, k)
        cand_ctx = self._safe_sample(candidate, k)
        return _format_context_block(seed, seed_ctx, candidate, cand_ctx, max_chars=self.args.snippet_max_chars)

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
                    return False, {"schema_used": "?", "unknown_terms": [], "duplicates": []}, ms, str(e)
                sleep_s = delay * (2 ** (attempt - 1)) * (1.0 + 0.25 * random.random())
                time.sleep(min(sleep_s, 5.0))
        return False, {"schema_used": "?", "unknown_terms": [], "duplicates": []}, 0, "unknown"

    def eval_pair(self, seed: str, candidate: str) -> DecisionRecord:
        # Basic validation
        if not isinstance(seed, str) or not isinstance(candidate, str) or not seed or not candidate:
            return DecisionRecord(seed=str(seed), candidate=str(candidate), accepted=False, decision="error",
                                  prompt_type="single", error="invalid seed/candidate type or empty string")

        key = (seed, candidate)
        if self.args.cache_path and key in self.cache:
            return self.cache[key]

        # Auto-accept shortcut when identical
        if self.args.auto_accept_if_equal and candidate == seed:
            rec = DecisionRecord(
                seed=seed, candidate=candidate, accepted=True, decision="accept",
                prompt_type="single", schema_used="-", attempts=0, latency_ms=0,
                relation="synonym" if self.args.relation_mode else None,
            )
            if self.args.cache_path:
                self._append_cache(self.args.cache_path, rec)
            return rec

        # Build corpus context block (3 longest by default)
        ctx_block = self._build_ctx_block(seed, candidate) if self.args.include_context else None

        ok, ver, ms, err = self._call_with_retries(self.judge.judge_single, seed, candidate, ctx_block)

        if err is not None:
            rec = DecisionRecord(
                seed=seed, candidate=candidate, accepted=False, decision="error",
                prompt_type="single", schema_used=ver.get("schema_used", "unknown"),
                unknown_terms=ver.get("unknown_terms", []), duplicates=ver.get("duplicates", []),
                attempts=max(1, int(self.args.retries) + 1), latency_ms=ms, error=str(err),
                relation=ver.get("relation"),
            )
        else:
            schema_ok = ver.get("schema_ok", True)
            if ok:
                decision = "accept"
            elif (not schema_ok) or ver.get("unknown_terms") or ver.get("duplicates"):
                decision = "unknown_mismatch"
            else:
                decision = "reject"
            rec = DecisionRecord(
                seed=seed, candidate=candidate, accepted=ok, decision=decision,
                prompt_type="single", schema_used=ver.get("schema_used", "unknown"),
                unknown_terms=ver.get("unknown_terms", []), duplicates=ver.get("duplicates", []),
                attempts=1, latency_ms=ms, error=None, relation=ver.get("relation"),
            )
        if self.args.cache_path:
            self._append_cache(self.args.cache_path, rec)
        return rec

    def process_seed(self, seed: str, candidates: List[str]) -> Tuple[Set[str], List[DecisionRecord], List[str]]:
        accepted: Set[str] = set()
        records: List[DecisionRecord] = []
        aligned: List[str] = []

        for cand in candidates:
            if not isinstance(cand, str) or not cand:
                records.append(DecisionRecord(seed=seed, candidate=str(cand), accepted=False, decision="error",
                                              prompt_type="single", error="non-string or empty candidate"))
                aligned.append("")
                continue

            rec = self.eval_pair(seed, cand)
            records.append(rec)
            if rec.accepted:
                accepted.add(cand)
                aligned.append(cand)
            else:
                aligned.append("")
        return accepted, records, aligned

# -----------------------------
# File I/O & Logging
# -----------------------------

def save_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("single_term_eval")
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

def write_seed_ndjson(path: Path, per_seed_rows: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in per_seed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def percentile(arr: List[int], p: float) -> float:
    if not arr:
        return 0.0
    arr = sorted(arr)
    k = (len(arr) - 1) * p / 100.0
    from math import floor, ceil
    f = floor(k)
    c = ceil(k)
    if f == c:
        return float(arr[int(k)])
    return arr[f] * (c - k) + arr[c] * (k - f)

def log_summary(logger: logging.Logger, all_records: List[DecisionRecord], accepted_by_seed: Dict[str, List[str]]):
    total = len(all_records)
    acc = sum(1 for r in all_records if r.decision == "accept")
    rej = sum(1 for r in all_records if r.decision == "reject")
    unk = sum(1 for r in all_records if r.decision == "unknown_mismatch")
    err = sum(1 for r in all_records if r.decision == "error")

    logger.info("# Single-term LLM evaluation summary (LLM-only relations, single-prompt)")
    logger.info(f"Total evals: {total} | accept={acc} reject={rej} unknown_mismatch={unk} error={err}")
    lat_ok = [r.latency_ms for r in all_records if r.decision in {"accept", "reject", "unknown_mismatch"}]
    if lat_ok:
        logger.info(f"Latency (ms) median={int(percentile(lat_ok,50))} p90={int(percentile(lat_ok,90))} max={max(lat_ok)}")
    logger.info("")

    if any(r.relation for r in all_records):
        from collections import Counter
        logger.info("## relation labels (when --relation_mode is on)")
        rel_counts = Counter(r.relation for r in all_records if r.relation)
        if rel_counts:
            pairs = ", ".join(f"{k}:{v}" for k, v in sorted(rel_counts.items(), key=lambda kv: (-kv[1], kv[0])))
            logger.info(pairs)
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
        logger.info(f"{seed:<28s} | total={len(seed_recs):4d} accept={acc_s:4d} reject={rej_s:4d} unknown={unk_s:3d} error={err_s:3d}")
    logger.info("")

    logger.info("## verification issues (model must return [] or [<exact candidate>])")
    issues = [r for r in all_records if r.unknown_terms or r.duplicates]
    if not issues:
        logger.info("(none)")
    else:
        for r in issues[:200]:
            unk = ",".join(r.unknown_terms) if r.unknown_terms else "-"
            dup = ",".join(r.duplicates) if r.duplicates else "-"
            logger.info(f"seed={r.seed} cand={r.candidate} schema={r.schema_used} | unknown={unk} dup={dup}")
    logger.info("")

    logger.info("## request/parse errors")
    errs = [r for r in all_records if r.decision == "error"]
    if not errs:
        logger.info("(none)")
    else:
        for r in errs[:200]:
            logger.info(f"seed={r.seed} cand={r.candidate} attempts={r.attempts} ms={r.latency_ms} | error={r.error}")

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Single-term LLM judgments (STRICT schema, R-CPD aware) with corpus context.")
    ap.add_argument("--expansions", default='testing/llm_addition_testing/sample/random_sample.json', help="Path to expansions JSON {seed: [terms]}")
    ap.add_argument("--outdir", required=False, help="Directory for outputs (ignored in debug mode)")

    # LLM
    ap.add_argument("--model", type=str, default="llama3.3:latest", help="Ollama model name")
    ap.add_argument("--url", type=str, default="http://localhost:11434/api/chat", help="Ollama chat endpoint")
    ap.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    ap.add_argument("--tokens", type=int, default=8196, help="LLM context tokens")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    ap.add_argument("--global_context", type=str, default=GLOBAL_CONTEXT_DEFAULT,
                    help="Short domain context sentence(s) injected into prompts")

    # Execution controls
    ap.add_argument("--retries", type=int, default=2, help="Retry count on request/parse errors")
    ap.add_argument("--sort", action="store_true", help="Sort final lists alphabetically where applicable")
    ap.add_argument("--auto_accept_if_equal", action="store_true", help="Auto-accept when candidate == seed")

    # Caching
    ap.add_argument("--cache_path", type=str, default=None, help="Optional JSONL cache file for decisions (read+append)")

    # Relation label mode (optional)
    ap.add_argument("--relation_mode", action="store_true",
                    help="If set, the model may also output a 'relation' label (hypernym/hyponym/holonym/cohyponym/…).")

    # Debug
    ap.add_argument("--debug_seed", type=str, default=None, help="If set, evaluate only this seed and --debug_term")
    ap.add_argument("--debug_term", type=str, default=None, help="If set with --debug_seed, evaluate only this candidate")

    # Corpus context
    ap.add_argument("--include_context", action="store_true", help="Include up to k corpus snippets for seed & candidate.")
    ap.add_argument("--ngram-dir", type=str, default="testing/ngram_evals_test_no_digits/4", help="Phrasers directory (passed to build_ngram_df)")
    ap.add_argument("--context-k", type=int, default=3, help="Snippets per term to include (longest first)")
    ap.add_argument("--snippet-max-chars", type=int, default=320, help="Max characters per snippet in prompt")

    args = ap.parse_args()

    expansions = load_json(args.expansions)
    if not isinstance(expansions, dict):
        raise ValueError("Expansions JSON must be an object mapping seeds to lists of terms.")

    # Build corpus DF if context requested
    df = build_ngram_df(args.ngram_dir) if args.include_context else None

    # --- DEBUG: single pair ---
    if args.debug_seed and args.debug_term:
        runner = SingleTermRunner(args, df=df, sample_docs_fn=sample_docs_containing)
        seed = args.debug_seed
        cand_list = expansions.get(seed)
        if not isinstance(cand_list, list):
            print(f"[warn] Seed '{seed}' not found or expansions not a list.")
            return
        if args.debug_term not in cand_list:
            print(f"[warn] Candidate '{args.debug_term}' is NOT in expansions for seed '{seed}'.")
        rec = runner.eval_pair(seed, args.debug_term)
        print("=== DEBUG SINGLE-PAIR ===")
        print(json.dumps(asdict(rec), indent=2, ensure_ascii=False))
        return

    # --- NORMAL MODE ---
    if not args.outdir:
        raise ValueError("--outdir is required unless --debug_seed and --debug_term are set")

    outdir = Path(args.outdir).expanduser().resolve()
    timestamp = datetime.now().strftime("%m_%d_%H_%M")
    eval_dir = outdir / f"single_term_eval_{timestamp}"
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
    runner = SingleTermRunner(args, df=df, sample_docs_fn=sample_docs_containing)

    seeds = list(expansions.keys())
    jobs: List[Tuple[str, List[str]]] = []
    for seed in seeds:
        cands = expansions.get(seed, [])
        if not isinstance(cands, list):
            logger.info(f"[warn] Seed '{seed}' expansions not a list; treating as empty.")
            cands = []
        else:
            cands = [str(c) for c in cands]
        jobs.append((seed, cands))

    all_records: List[DecisionRecord] = []
    accepted_by_seed: Dict[str, List[str]] = {}
    aligned_by_seed: Dict[str, List[str]] = {}
    per_seed_rows: List[dict] = []

    logger.info("# Processing seeds sequentially.")
    for seed, cand_list in jobs:
        acc, recs, aligned = runner.process_seed(seed, cand_list)
        # Subset check + logging: ensure accepted ⊆ original expansions
        original_list = expansions.get(seed, [])
        original_set = set(original_list) if isinstance(original_list, list) else set()
        violations = sorted([t for t in acc if t not in original_set])
        if violations:
            acc = set([t for t in acc if t in original_set])
        acc_list = sorted(acc) if args.sort else list(acc)
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

    accepted_all_flat: List[str] = []
    for seed, _cands in jobs:
        aligned = aligned_by_seed.get(seed, [])
        for term in aligned:
            if term:
                accepted_all_flat.append(term)
    with open(accepted_all_flat_path, "w", encoding="utf-8") as f:
        json.dump(accepted_all_flat, f, indent=2, ensure_ascii=False)

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
        "relation_mode": bool(args.relation_mode),
        "include_context": bool(args.include_context),
        "context_k": int(args.context_k),
        "ngram_dir": args.ngram_dir,
    }
    save_json(summary_path, summary)

    logger.info(f"# Output dir: {eval_dir.resolve()}")
    logger.info(f"# Pair decisions : {ndjson_pairs_path.resolve()}")
    logger.info(f"# Seed rows      : {ndjson_seeds_path.resolve()}")
    logger.info(f"# Accepted map   : {accepted_by_seed_path.resolve()}")
    logger.info(f"# Filtered exp   : {filtered_expansions_path.resolve()}")
    logger.info(f"# Accepted flat  : {accepted_all_flat_path.resolve()}")
    logger.info(f"# Accepted align : {accepted_aligned_path.resolve()}")
    logger.info(f"# Summary        : {summary_path.resolve()}")
    logger.info(f"# Retries={args.retries} Sort={bool(args.sort)} RelationMode={bool(args.relation_mode)} "
                f"IncludeContext={bool(args.include_context)} ContextK={args.context_k}")
    logger.info("")
    log_summary(logger, all_records, accepted_by_seed)

if __name__ == "__main__":
    main()
