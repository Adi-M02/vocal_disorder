#!/usr/bin/env python3
"""
Single-term evaluator for seed-term expansions (STRICT one-candidate schema)
===========================================================================

What this does
--------------
- Evaluates **one expansion term at a time** against its seed using an LLM.
- **Strict I/O schema per call** so the model can ONLY emit either [] (reject) or ["<exact candidate>"] (accept).
- Rich logging + verification:
  • Per-pair NDJSON audit: latency, attempts, schema used, errors, unknown returns.

  • Per-seed NDJSON line: {seed, accepted:[...], checked_subset:bool, violations:[...]}.

  • Human-readable summary log with per-seed acceptance rates + error sections.
- Outputs:
  • decisions.ndjson                (per-pair audit)
  • seeds_accepted.ndjson          (one line per seed with accepted list + subset check)
  • accepted_by_seed.json          { seed: [accepted terms] } (subset of input expansions)
  • filtered_expansions.json       { seed: [accepted terms] } (same as above; name mirrors input format)
  • accepted_all_flat.json         [all accepted terms across all seeds, in encounter order]
  • accepted_aligned_by_seed.json  { seed: [accepted-or-empty-string aligned to input order] }
  • summary.json                   run stats + settings

Speed knobs
-----------
- --concurrency: number of seeds in parallel
- --seed_concurrency: parallelism *within* a seed (disables anchors)
- --use_anchors: previously-accepted terms for a seed inform later decisions
- --cache_path: JSONL cache to reuse decisions across runs

Debug a single (seed, term):
  python augment_seeds_single_term.py \
      --expansions path/to/expansions.json \
      --model llama3.3:latest \
      --debug_seed nauseous --debug_term nauseate
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
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------
# Helpers
# -----------------------------

def norm(tok: str) -> str:
    return tok.strip().lower()


def join_uniq_sorted(iterable: Iterable[str]) -> str:
    return ", ".join(sorted(set(iterable)))


def verify_unchanged(candidates: List[str], returned_terms: List[str]) -> Dict[str, List[str]]:
    """Compare LLM-returned terms to the original candidates (exact string match).
    Returns dict with 'unknown_terms' and 'duplicates'.
    """
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
# LLM client (Ollama) — STRICT single-candidate protocol
# -----------------------------

class LlmSimilarityDecider:
    def __init__(
        self,
        model: str,
        url: str = "http://localhost:11434/api/chat",
        temperature: float = 0.0,
        timeout: int = 60,
        inclusion_bias: str = "lenient",  # 'normal' | 'lenient'
        global_context: str = (
            "The domain is R-CPD (Retrograde Cricopharyngeus Dysfunction: inability to burp) on Reddit. "
            "Expansions may include medical terminology, anatomy/physiology terms (e.g., cricopharyngeus, UES, esophagus), "
            "diagnostics (ENT visits, manometry, FEES, barium swallow, endoscopy), interventions/treatments "
            "(botox to cricopharyngeus/UES, dilation, therapy), related symptoms (chest pressure, bloating, gurgling, hiccups, "
            "nausea, reflux-like sensations), patient emotions (anxiety, embarrassment, frustration, relief, validation), "
            "actions/behaviors (massage, breathing techniques, carbonation tests, dietary changes, booking appointments), "
            "logistics (insurance, referrals, clinic names), and community/platform references (subreddit, reddit, tiktok, instagram, threads, groups)."
        ),
        session: Optional[requests.Session] = None,
    ):
        self.url = url
        self.timeout = timeout
        self.inclusion_bias = inclusion_bias.lower().strip()
        self.global_context = global_context.strip()
        self.headers = {"Content-Type": "application/json"}
        self.session = session or requests.Session()

        # NOTE: We no longer set a static 'format' in base_payload. Instead we
        # build a per-call schema locked to the exact candidate string.
        self.base_payload = {
            "model": model,
            "options": {"temperature": float(temperature)},
            "stream": False,
        }

        bias_line = (
            "Err slightly on the INCLUSIVE side when a candidate would help a user find/name/describe the same concept "
            "or closely neighboring concepts within this domain (medical terms, diagnostics, related symptoms, emotions, "
            "actions/behaviors, logistics, and community/platform terms)."
            if self.inclusion_bias == "lenient"
            else "Be reasonably inclusive across the listed categories, but avoid broad unrelated associations."
        )

        # System prompts explicitly state single-candidate contract and output form.
        self.system_initial = (
            "You are a semantic similarity decider for short terms and underscore-separated MWEs.
"
            f"DOMAIN CONTEXT:
{self.global_context}

"
            "TASK (STRICT):
"
            "Given a SEED and EXACTLY ONE CANDIDATE, return STRICT JSON with keys {seed, decisions}.
"
            "The ONLY valid outputs are either [] (reject) or [<exact candidate>] (accept).
"
            "Do NOT alter, normalize, or invent strings. Return no explanations.

"
            "CRITICAL:
"
            " • Consider text EXACTLY as given; do NOT modify (even r/noburp).
"
            " • Return STRICT JSON ONLY with keys: seed, decisions (array of strings).
"
            f" • {bias_line}
"
        )

        self.system_closure = (
            "You are expanding a concept bucket using ANCHORS (already accepted examples).
"
            "You will be given a SEED, a small ANCHORS list, and EXACTLY ONE REMAINING_CANDIDATE.
"
            "Return STRICT JSON {seed, decisions}. The ONLY valid outputs are [] or [<exact candidate>].
"
            "Choose [<candidate>] only if it clearly belongs with the anchors for the seed within the R-CPD domain.

"
            f"DOMAIN CONTEXT:
{self.global_context}
"
        )

    # Per-call JSON schema locked to the exact candidate
    def _build_single_candidate_schema(self, candidate: str) -> dict:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["seed", "decisions"],
            "properties": {
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
            },
        }

    def _build_user_prompt_initial(self, seed: str, candidate: str) -> str:
        return (
            f"SEED: {seed}
"
            f"CANDIDATE (evaluate only this exact string; accept ⇒ return it, reject ⇒ return []):
- {candidate}

"
            "Respond ONLY with JSON {\"seed\": <seed>, \"decisions\": [] or [<exact candidate>]}."
        )

    def _build_user_prompt_closure(self, seed: str, anchors: List[str], candidate: str) -> str:
        anc_str = "
".join(f"- {a}" for a in anchors) if anchors else "(none)"
        return (
            f"SEED: {seed}
"
            f"ANCHORS (examples already accepted for this seed):
{anc_str}

"
            f"REMAINING_CANDIDATE (evaluate ONLY this exact string):
- {candidate}

"
            "Respond ONLY with JSON {\"seed\": <seed>, \"decisions\": [] or [<exact candidate>]}."
        )

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
            # Extremely defensive: convert list of objects with {term: ...} into strings
            terms: List[str] = []
            for item in raw:
                t = item.get("term") if isinstance(item, dict) else None
                if isinstance(t, str):
                    terms.append(t)
            return terms, "object->string"
        else:
            return [t for t in raw if isinstance(t, str)], "string"

    # Single-candidate calls ----------------------------------------------
    def judge_single_initial(self, seed: str, candidate: str) -> Tuple[bool, Dict]:
        payload = dict(self.base_payload)
        payload["format"] = self._build_single_candidate_schema(candidate)
        payload["messages"] = [
            {"role": "system", "content": self.system_initial},
            {"role": "user", "content": self._build_user_prompt_initial(seed, candidate)},
        ]
        out = self._post(payload)
        terms, schema_used = self._coerce_terms(out)
        ver = verify_unchanged([candidate], terms)
        ver["schema_used"] = schema_used
        ver["phase"] = "initial"
        # Extra guard in case format is ignored by backend
        valid = (candidate in terms) or (len(terms) == 0)
        return (valid and not ver["unknown_terms"] and len(terms) <= 1), ver

    def judge_single_with_anchors(self, seed: str, anchors: List[str], candidate: str) -> Tuple[bool, Dict]:
        payload = dict(self.base_payload)
        payload["format"] = self._build_single_candidate_schema(candidate)
        payload["messages"] = [
            {"role": "system", "content": self.system_closure},
            {"role": "user", "content": self._build_user_prompt_closure(seed, anchors, candidate)},
        ]
        out = self._post(payload)
        terms, schema_used = self._coerce_terms(out)
        ver = verify_unchanged([candidate], terms)
        ver["schema_used"] = schema_used
        ver["phase"] = "closure"
        valid = (candidate in terms) or (len(terms) == 0)
        return (valid and not ver["unknown_terms"] and len(terms) <= 1), ver


# -----------------------------
# Data model for per-term decisions
# -----------------------------

@dataclass
class DecisionRecord:
    seed: str
    candidate: str
    accepted: bool
    decision: str  # "accept" | "reject" | "error" | "unknown_mismatch"
    prompt_type: str  # "initial" | "closure" | "shortcut"
    schema_used: str = "unknown"
    unknown_terms: List[str] = field(default_factory=list)
    duplicates: List[str] = field(default_factory=list)
    attempts: int = 1
    latency_ms: int = 0
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# -----------------------------
# Runner
# -----------------------------

class SingleTermRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.session = requests.Session()
        self.judge = LlmSimilarityDecider(
            model=args.model,
            url=args.url,
            temperature=args.temperature,
            timeout=args.timeout,
            inclusion_bias=args.inclusion_bias,
            global_context=args.global_context,
            session=self.session,
        )
        self.cache: Dict[Tuple[str, str], DecisionRecord] = {}
        if args.cache_path and Path(args.cache_path).exists():
            self._load_cache(args.cache_path)

    # Cache ---------------------------------------------------------------
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
                f.write(rec.to_json() + "
")
        except Exception as e:
            print(f"[warn] Failed to write cache {path}: {e}")

    # Core eval -----------------------------------------------------------
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
                    return False, {"phase": "?", "schema_used": "?", "unknown_terms": [], "duplicates": []}, ms, str(e)
                # Exponential backoff with jitter
                sleep_s = delay * (2 ** (attempt - 1)) * (1.0 + 0.25 * random.random())
                time.sleep(min(sleep_s, 5.0))
        return False, {"phase": "?", "schema_used": "?", "unknown_terms": [], "duplicates": []}, 0, "unknown"

    def eval_pair(self, seed: str, candidate: str, anchors: Optional[List[str]] = None) -> DecisionRecord:
        # Basic input validation
        if not isinstance(seed, str) or not isinstance(candidate, str) or not seed or not candidate:
            return DecisionRecord(
                seed=str(seed),
                candidate=str(candidate),
                accepted=False,
                decision="error",
                prompt_type="initial",
                error="invalid seed/candidate type or empty string",
            )

        key = (seed, candidate)
        if self.args.cache_path and key in self.cache:
            return self.cache[key]

        # Auto-accept shortcut when candidate == seed
        if self.args.auto_accept_if_equal and candidate == seed:
            rec = DecisionRecord(
                seed=seed,
                candidate=candidate,
                accepted=True,
                decision="accept",
                prompt_type="shortcut",
                schema_used="-",
                attempts=0,
                latency_ms=0,
            )
            if self.args.cache_path:
                self._append_cache(self.args.cache_path, rec)
            return rec

        use_anchors = bool(self.args.use_anchors and anchors)
        if use_anchors:
            ok, ver, ms, err = self._call_with_retries(self.judge.judge_single_with_anchors, seed, anchors or [], candidate)
            prompt_type = "closure"
        else:
            ok, ver, ms, err = self._call_with_retries(self.judge.judge_single_initial, seed, candidate)
            prompt_type = "initial"

        if err is not None:
            rec = DecisionRecord(
                seed=seed,
                candidate=candidate,
                accepted=False,
                decision="error",
                prompt_type=prompt_type,
                schema_used=ver.get("schema_used", "unknown"),
                unknown_terms=ver.get("unknown_terms", []),
                duplicates=ver.get("duplicates", []),
                attempts=max(1, int(self.args.retries) + 1),
                latency_ms=ms,
                error=str(err),
            )
        else:
            decision = "accept" if ok else ("unknown_mismatch" if ver.get("unknown_terms") or ver.get("duplicates") else "reject")
            rec = DecisionRecord(
                seed=seed,
                candidate=candidate,
                accepted=ok,
                decision=decision,
                prompt_type=prompt_type,
                schema_used=ver.get("schema_used", "unknown"),
                unknown_terms=ver.get("unknown_terms", []),
                duplicates=ver.get("duplicates", []),
                attempts=1,
                latency_ms=ms,
                error=None,
            )
        if self.args.cache_path:
            self._append_cache(self.args.cache_path, rec)
        return rec

    # Per-seed sequential flow so anchors can grow ------------------------
    def process_seed_sequential(self, seed: str, candidates: List[str]) -> Tuple[Set[str], List[DecisionRecord], List[str]]:
        anchors: List[str] = []
        accepted: Set[str] = set()
        records: List[DecisionRecord] = []
        aligned: List[str] = []  # same length/order as input candidates; accepted term or ""
        for cand in candidates:
            if not isinstance(cand, str) or not cand:
                # Emit an error record for bad candidate type
                records.append(DecisionRecord(seed=seed, candidate=str(cand), accepted=False, decision="error", prompt_type="initial", error="non-string or empty candidate"))
                aligned.append("")
                continue
            rec = self.eval_pair(seed, cand, anchors=anchors)
            records.append(rec)
            if rec.accepted:
                accepted.add(cand)
                if self.args.use_anchors:
                    anchors.append(cand)
                aligned.append(cand)
            else:
                aligned.append("")
        return accepted, records, aligned

    # Per-seed parallel flow (no anchors) ---------------------------------
    def process_seed_parallel(self, seed: str, candidates: List[str]) -> Tuple[Set[str], List[DecisionRecord], List[str]]:
        accepted: Set[str] = set()
        records: List[DecisionRecord] = []
        with ThreadPoolExecutor(max_workers=self.args.seed_concurrency) as pool:
            fut_to_idx = {pool.submit(self.eval_pair, seed, cand, None): i for i, cand in enumerate(candidates)}
            results: List[Tuple[int, DecisionRecord]] = []
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                rec = fut.result()
                results.append((idx, rec))
                if rec.accepted:
                    accepted.add(rec.candidate)
        # Restore original order and build aligned list
        results.sort(key=lambda x: x[0])
        aligned: List[str] = []
        for idx, rec in results:
            records.append(rec)
            aligned.append(rec.candidate if rec.accepted else "")
        return accepted, records, aligned


# -----------------------------
# File I/O & Logging
# -----------------------------

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
            f.write(r.to_json() + "
")


def write_seed_ndjson(path: Path, per_seed_rows: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in per_seed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "
")


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


def log_summary(logger: logging.Logger, all_records: List[DecisionRecord], accepted_by_seed: Dict[str, List[str]]):
    total = len(all_records)
    acc = sum(1 for r in all_records if r.decision == "accept")
    rej = sum(1 for r in all_records if r.decision == "reject")
    unk = sum(1 for r in all_records if r.decision == "unknown_mismatch")
    err = sum(1 for r in all_records if r.decision == "error")

    logger.info("# Single-term LLM evaluation summary")
    logger.info(f"Total evals: {total} | accept={acc} reject={rej} unknown_mismatch={unk} error={err}")
    lat_ok = [r.latency_ms for r in all_records if r.decision in {"accept", "reject", "unknown_mismatch"}]
    if lat_ok:
        logger.info(
            f"Latency (ms) median={int(percentile(lat_ok,50))} p90={int(percentile(lat_ok,90))} max={max(lat_ok)}"
        )
    logger.info("")

    # Per-seed acceptance rates
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
    logger.info("")

    # Verification issues
    logger.info("## verification issues (model must return [] or [<exact candidate>])")
    issues = [r for r in all_records if r.unknown_terms or r.duplicates]
    if not issues:
        logger.info("(none)")
    else:
        for r in issues[:200]:  # cap printed lines
            unk = ",".join(r.unknown_terms) if r.unknown_terms else "-"
            dup = ",".join(r.duplicates) if r.duplicates else "-"
            logger.info(
                f"seed={r.seed} cand={r.candidate} phase={r.prompt_type} schema={r.schema_used} | unknown={unk} dup={dup}"
            )
    logger.info("")

    # Errors
    logger.info("## request/parse errors")
    errs = [r for r in all_records if r.decision == "error"]
    if not errs:
        logger.info("(none)")
    else:
        for r in errs[:200]:
            logger.info(
                f"seed={r.seed} cand={r.candidate} phase={r.prompt_type} attempts={r.attempts} ms={r.latency_ms} | error={r.error}"
            )


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Single-term LLM judgments for seed expansions (STRICT schema, R-CPD aware).")
    ap.add_argument("--expansions", required=True, help="Path to expansions JSON {seed: [terms]}")
    ap.add_argument("--outdir", help="Directory for outputs (ignored in debug mode)")

    # LLM
    ap.add_argument("--model", type=str, default="llama3.3:latest", help="Ollama model name")
    ap.add_argument("--url", type=str, default="http://localhost:11434/api/chat", help="Ollama chat endpoint")
    ap.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    ap.add_argument("--inclusion_bias", type=str, choices=["normal", "lenient"], default="lenient")
    ap.add_argument("--global_context", type=str, default=(
        "The domain is R-CPD (Retrograde Cricopharyngeus Dysfunction: inability to burp) on Reddit. "
        "Expansions may include medical terminology, anatomy/physiology terms (e.g., cricopharyngeus, UES, esophagus), "
        "diagnostics (ENT visits, manometry, FEES, barium swallow, endoscopy), interventions/treatments (botox to cricopharyngeus/UES, dilation, therapy), "
        "related symptoms (chest pressure, bloating, gurgling, hiccups, nausea, reflux-like sensations), patient emotions (anxiety, embarrassment, frustration, relief, validation), "
        "actions/behaviors (massage, breathing techniques, carbonation tests, dietary changes, booking appointments), logistics (insurance, referrals, clinic names), and community/platform references."),
        help="Short domain context sentence(s) injected into prompts",
    )

    # Execution controls
    ap.add_argument("--concurrency", type=int, default=6, help="Max seeds processed in parallel")
    ap.add_argument("--seed_concurrency", type=int, default=1, help="Parallelism *within* a seed (disables anchors if >1)")
    ap.add_argument("--use_anchors", action="store_true", help="Within a seed, feed accepted terms as anchors for subsequent evaluations")
    ap.add_argument("--retries", type=int, default=2, help="Retry count on request/parse errors (exponential backoff)")
    ap.add_argument("--sort", action="store_true", help="Sort final lists alphabetically where applicable")
    ap.add_argument("--auto_accept_if_equal", action="store_true", help="Auto-accept when candidate == seed")

    # Filtering / limits
    ap.add_argument("--seed_filter", type=str, nargs="*", default=None, help="Only process these seeds (exact match)")
    ap.add_argument("--limit_per_seed", type=int, default=0, help="Limit number of candidates per seed (0=all)")

    # Caching
    ap.add_argument("--cache_path", type=str, default=None, help="Optional JSONL cache file for decisions (read+append)")

    # Debug
    ap.add_argument("--debug_seed", type=str, default=None, help="If set, evaluate only this seed and --debug_term")
    ap.add_argument("--debug_term", type=str, default=None, help="If set with --debug_seed, evaluate only this candidate")

    args = ap.parse_args()

    expansions = load_json(args.expansions)
    if not isinstance(expansions, dict):
        raise ValueError("Expansions JSON must be an object mapping seeds to lists of terms.")

    # --- DEBUG MODE: single pair ---
    if args.debug_seed and args.debug_term:
        runner = SingleTermRunner(args)
        seed = args.debug_seed
        cand_list = expansions.get(seed)
        if not isinstance(cand_list, list):
            print(f"[warn] Seed '{seed}' not found or expansions not a list.")
            return
        if args.debug_term not in cand_list:
            print(f"[warn] Candidate '{args.debug_term}' is NOT in expansions for seed '{seed}'.")
        rec = runner.eval_pair(seed, args.debug_term, anchors=None)
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
    filtered_expansions_path = eval_dir / "filtered_expansions.json"  # same shape as input
    accepted_all_flat_path = eval_dir / "accepted_all_flat.json"
    accepted_aligned_path = eval_dir / "accepted_aligned_by_seed.json"
    summary_path = eval_dir / "summary.json"

    logger = setup_logger(log_path)
    runner = SingleTermRunner(args)

    # Filter seeds if requested
    seeds = list(expansions.keys())
    if args.seed_filter:
        seed_set = set(args.seed_filter)
        seeds = [s for s in seeds if s in seed_set]

    # Build job list
    jobs: List[Tuple[str, List[str]]] = []
    for seed in seeds:
        cands = expansions.get(seed, [])
        if not isinstance(cands, list):
            # Log and continue (but we still record an empty accepted list for consistency)
            logger.info(f"[warn] Seed '{seed}' expansions not a list; treating as empty.")
            cands = []
        else:
            # Coerce to strings defensively
            cands = [str(c) for c in cands]
        if args.limit_per_seed and args.limit_per_seed > 0:
            cands = cands[: args.limit_per_seed]
        jobs.append((seed, cands))

    all_records: List[DecisionRecord] = []
    accepted_by_seed: Dict[str, List[str]] = {}
    aligned_by_seed: Dict[str, List[str]] = {}
    per_seed_rows: List[dict] = []

    per_seed_parallel = args.seed_concurrency > 1

    def run_one(seed: str, cand_list: List[str]) -> Tuple[str, Set[str], List[DecisionRecord], List[str]]:
        if per_seed_parallel:
            acc, recs, aligned = runner.process_seed_parallel(seed, cand_list)
        else:
            acc, recs, aligned = runner.process_seed_sequential(seed, cand_list)
        return seed, acc, recs, aligned

    # Parallelize across seeds
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futs = [pool.submit(run_one, seed, cands) for seed, cands in jobs]
        for fut in as_completed(futs):
            seed, acc, recs, aligned = fut.result()
            # Subset check + logging: make sure accepted ⊆ original expansions
            original_list = expansions.get(seed, [])
            original_set = set(original_list) if isinstance(original_list, list) else set()
            violations = sorted([t for t in acc if t not in original_set])
            if violations:
                # Drop violators from persisted outputs, but log them.
                acc = set([t for t in acc if t in original_set])
            acc_list = sorted(acc) if args.sort else list(acc)
            accepted_by_seed[seed] = acc_list
            aligned_by_seed[seed] = aligned  # already aligned to input order; "" for rejects
            per_seed_rows.append({
                "seed": seed,
                "accepted": acc_list,
                "checked_subset": len(violations) == 0,
                "violations": violations,
            })
            all_records.extend(recs)

    # Write per-pair audit
    write_ndjson(ndjson_pairs_path, all_records)

    # Write per-seed NDJSON rows
    write_seed_ndjson(ndjson_seeds_path, per_seed_rows)

    # Write mapping seed -> accepted list (filtered) — identical shape to input
    save_json(accepted_by_seed_path, accepted_by_seed)
    save_json(filtered_expansions_path, accepted_by_seed)

    # Write aligned mapping seed -> list with same length as input (accepted term or "")
    save_json(accepted_aligned_path, aligned_by_seed)

    # Build flat list of all accepted terms across seeds (encounter order from per_seed_rows)
    accepted_all_flat: List[str] = []
    for row in per_seed_rows:
        accepted_all_flat.extend(row["accepted"])
    save_json(accepted_all_flat_path, accepted_all_flat)

    # Summary JSON
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
        "timeout": args.timeout,
        "concurrency": args.concurrency,
        "seed_concurrency": args.seed_concurrency,
        "use_anchors": bool(args.use_anchors and not per_seed_parallel),
        "retries": args.retries,
        "sorted": bool(args.sort),
    }
    save_json(summary_path, summary)

    # Human-readable log summary
    logger.info(f"# Output dir: {eval_dir.resolve()}")
    logger.info(f"# Pair decisions : {ndjson_pairs_path.resolve()}")
    logger.info(f"# Seed rows      : {ndjson_seeds_path.resolve()}")
    logger.info(f"# Accepted map   : {accepted_by_seed_path.resolve()}")
    logger.info(f"# Filtered exp   : {filtered_expansions_path.resolve()}")
    logger.info(f"# Accepted flat  : {accepted_all_flat_path.resolve()}")
    logger.info(f"# Accepted align : {accepted_aligned_path.resolve()}")
    logger.info(f"# Summary        : {summary_path.resolve()}")
    logger.info(
        f"# Model={args.model} URL={args.url} Temp={args.temperature} Timeout={args.timeout}s Concurrency={args.concurrency}/seed={args.seed_concurrency}"
    )
    logger.info(f"# UseAnchors={bool(args.use_anchors and not per_seed_parallel)} Retries={args.retries} Sort={bool(args.sort)}")
    logger.info("")
    log_summary(logger, all_records, accepted_by_seed)


if __name__ == "__main__":
    main()
