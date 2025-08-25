#!/usr/bin/env python3
"""
Batch evaluator for seed-term expansions (LLM-only WordNet-like relations, single call per seed)
================================================================================================

What this does
--------------
- For each SEED, sends the **entire candidate list** to the LLM in **one request**.
- No WordNet library: the LLM internally decides relations akin to WordNet:
  hypernym (umbrella), hyponym (specific), holonym (whole), cohyponym (sibling),
  plus synonyms / near-synonyms / morphological variants — all within the R-CPD domain.
- Supports optional anchors: with `--use_anchors` we pass the FULL set of seed terms
  as contextual anchors (mirrors your earlier closure idea).
- Optional **iterative closure** inside the candidate list up to `--closure_iters` rounds
  (OFF by default so single vs batch are directly comparable).
- **STRICT output contract**: the model must return JSON:
    {
      "seed": "<echo seed>",
      "decisions": ["subset of provided candidates" ...],
      // optional when --relation_mode:
      "relations": { "<accepted_cand>": "<relation_label>", ... }
    }
  relation labels from: hypernym, hyponym, holonym, cohyponym,
  synonym, morphological_variant, near_synonym, unrelated, unknown.

Outputs (same set as single-term)
---------------------------------
  • decisions.ndjson
  • seeds_accepted.ndjson
  • accepted_by_seed.json
  • filtered_expansions.json
  • accepted_all_flat.json
  • accepted_aligned_by_seed.json
  • summary.json
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

# -----------------------------
# Shared defaults (identical to single)
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
# Relation categories
# -----------------------------

RELATION_ENUM = [
    "hypernym",
    "hyponym",
    "holonym",
    "cohyponym",
    "synonym",
    "morphological_variant",
    "near_synonym",
    "unrelated",
    "unknown",
]

# -----------------------------
# LLM client (Ollama) — STRICT batch protocol (one call per seed)
# -----------------------------

class LlmBatchDecider:
    def __init__(
        self,
        model: str,
        url: str = "http://localhost:11434/api/chat",
        temperature: float = 0.0,
        tokens = 2048,
        timeout: int = 60,
        include_relations: bool = False,
        closure_iters: int = 0,  # default OFF for parity with single-term
        global_context: str = GLOBAL_CONTEXT_DEFAULT,
        session: Optional[requests.Session] = None,
    ):
        self.url = url
        self.timeout = timeout
        self.global_context = (global_context or GLOBAL_CONTEXT_DEFAULT).strip()
        self.include_relations = bool(include_relations)
        self.closure_iters = max(0, int(closure_iters))
        self.session = session or requests.Session()

        self.base_payload = {
            "model": model,
            "options": {"temperature": float(temperature), "num_ctx": tokens},
            "stream": False,
        }

        # Batch system message (no anchors)
        closure_blurb = (
            "Iterative closure: perform up to N rounds (N provided) of closure within the candidate list: "
            "if a candidate is accepted and another candidate stands in one of the above relations to it (or to the seed), "
            "accept that other candidate as well.\n"
            if self.closure_iters > 0 else ""
        )

        self.system_batch = (
            "You are a semantic decider for short terms / underscore-separated MWEs in the R-CPD domain.\n"
            f"DOMAIN CONTEXT:\n{self.global_context}\n\n"
            "TASK (STRICT):\n"
            "You will be given a SEED and a CANDIDATE_LIST. Decide which candidates belong with the seed.\n"
            "Accept a candidate if it helps a user find/name/describe the same concept as the seed OR is a closely neighboring\n"
            "concept in this domain via ANY of these relations:\n"
            "  • hypernym (umbrella term of the seed)\n"
            "  • hyponym (specific instance/type of the seed)\n"
            "  • holonym (a whole that includes the seed as part/member)\n"
            "  • cohyponym (sibling under the same umbrella as the seed)\n"
            "Additionally accept synonyms, near-synonyms, and morphological variants (noun/verb/adjective forms of the same phenomenon).\n"
            f"{closure_blurb}"
            "STRICT OUTPUT:\n"
            "Return ONLY JSON with keys:\n"
            "  seed: string (exact echo of the input SEED)\n"
            "  decisions: array of strings (subset of EXACT items from CANDIDATE_LIST; no extra items; no duplicates)\n"
            + ("  relations: object {<accepted_candidate>: <one-of enum>} (optional map; keys must be a subset of 'decisions')\n" if self.include_relations else "") +
            "Do not include explanations or commentary."
        )

        # Batch system message (with anchors)
        self.system_batch_with_anchors = (
            "You are a semantic decider for short terms / underscore-separated MWEs in the R-CPD domain.\n"
            f"DOMAIN CONTEXT:\n{self.global_context}\n\n"
            "You will be given:\n"
            "  • SEED (the focal term),\n"
            "  • VOCAB_BUCKET_SEEDS (a long list of known seed terms as anchors/context),\n"
            "  • CANDIDATE_LIST.\n"
            "Decide which CANDIDATE_LIST items belong with the seed within the R-CPD domain using the anchors as context.\n"
            "Accept if the candidate relates to the seed (or to an already accepted candidate) via ANY of:\n"
            "  hypernym, hyponym, holonym, cohyponym, synonym, near_synonym, morphological_variant.\n"
            f"{closure_blurb}"
            "STRICT OUTPUT:\n"
            "Return ONLY JSON with keys:\n"
            "  seed: string (exact echo of the input SEED)\n"
            "  decisions: array of strings (subset of EXACT items from CANDIDATE_LIST; no extra items; no duplicates)\n"
            + ("  relations: object {<accepted_candidate>: <one-of enum>} (optional map; keys must be a subset of 'decisions')\n" if self.include_relations else "") +
            "Do not include explanations or commentary."
        )

    # JSON schema for batch: restrict decisions to provided candidates; optional relations map
    def _build_batch_schema(self, candidates: List[str]) -> dict:
        props = {
            "seed": {"type": "string"},
            "decisions": {
                "type": "array",
                "items": {"type": "string", "enum": candidates},
                "uniqueItems": True,
            },
        }
        required = ["seed", "decisions"]

        if self.include_relations:
            props["relations"] = {
                "type": "object",
                "propertyNames": {"type": "string", "enum": candidates},
                "additionalProperties": {"type": "string", "enum": RELATION_ENUM},
            }

        return {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": props,
        }

    def _build_user_prompt(self, seed: str, candidates: List[str], anchors: Optional[List[str]]) -> str:
        cand_str = "\n".join(f"- {c}" for c in candidates) if candidates else "(none)"
        header = f"SEED: {seed}\n"
        if anchors:
            anc_str = "\n".join(f"- {a}" for a in anchors)
            header += f"VOCAB_BUCKET_SEEDS (anchors/context):\n{anc_str}\n\n"
        body = f"CANDIDATE_LIST (evaluate ONLY these exact strings):\n{cand_str}\n\n"
        if self.closure_iters > 0:
            body += f"ITERATIVE_CLOSURE_ROUNDS: {self.closure_iters}\n\n"
        tail = (
            "Respond ONLY with JSON: {\"seed\": <seed>, \"decisions\": [<subset of candidates>]}"
            if not self.include_relations
            else "Respond ONLY with JSON: {\"seed\": <seed>, \"decisions\": [<subset>], \"relations\": {<accepted_candidate>: <one of "
                 + ", ".join(RELATION_ENUM) + ">}}"
        )
        return header + body + tail

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

    def judge_batch(self, seed: str, candidates: List[str], anchors: Optional[List[str]]) -> Tuple[Set[str], Dict]:
        payload = dict(self.base_payload)
        payload["format"] = self._build_batch_schema(candidates)
        # Choose system message variant
        system_msg = self.system_batch_with_anchors if anchors else self.system_batch
        payload["messages"] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": self._build_user_prompt(seed, candidates, anchors)},
        ]
        out = self._post(payload)

        # Parse outputs
        decisions_raw = out.get("decisions", []) if isinstance(out, dict) else []
        relations_raw = out.get("relations", {}) if (self.include_relations and isinstance(out, dict)) else {}
        if not isinstance(decisions_raw, list):
            decisions_raw = []
        if not isinstance(relations_raw, dict):
            relations_raw = {}

        accepted_terms = [t for t in decisions_raw if isinstance(t, str)]
        ver = verify_unchanged(candidates, accepted_terms)
        returned_seed = out.get("seed") if isinstance(out, dict) else None

        # Constrain relations map
        relations: Dict[str, str] = {}
        for k, v in relations_raw.items():
            if isinstance(k, str) and isinstance(v, str) and k in accepted_terms and v in RELATION_ENUM:
                relations[k] = v

        schema_used = "batch"
        ver.update({
            "schema_used": schema_used,
            "phase": "batch_with_anchors" if anchors else "batch",
            "seed_echo_ok": (returned_seed == seed),
            "relations": relations,
        })

        # schema_ok if: seed echo ok, no unknown/dupes, and relations keys subset of accepted
        rel_keys_ok = all(k in accepted_terms for k in relations.keys())
        schema_ok = ver["seed_echo_ok"] and not ver["unknown_terms"] and not ver["duplicates"] and rel_keys_ok
        ver["schema_ok"] = schema_ok

        return set(accepted_terms), ver

# -----------------------------
# Data model for per-term decisions (expanded from batch)
# -----------------------------

@dataclass
class DecisionRecord:
    seed: str
    candidate: str
    accepted: bool
    decision: str  # "accept" | "reject" | "error" | "unknown_mismatch"
    prompt_type: str  # "batch" | "batch_with_anchors" | "shortcut"
    schema_used: str = "batch"
    unknown_terms: List[str] = field(default_factory=list)
    duplicates: List[str] = field(default_factory=list)
    attempts: int = 1
    latency_ms: int = 0
    error: Optional[str] = None
    # Optional relation label (when --relation_mode is on)
    relation: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

# -----------------------------
# Runner (sequential across seeds; one LLM call per seed)
# -----------------------------

class BatchRunner:
    def __init__(self, args: argparse.Namespace, global_anchors: Optional[List[str]] = None):
        self.args = args
        self.session = requests.Session()
        self.judge = LlmBatchDecider(
            model=args.model,
            url=args.url,
            temperature=args.temperature,
            tokens=args.tokens,
            timeout=args.timeout,
            include_relations=bool(args.relation_mode),
            closure_iters=int(args.closure_iters),
            global_context=args.global_context,
            session=self.session,
        )
        self.global_anchors: List[str] = list(global_anchors or [])
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
                f.write(rec.to_json() + "\n")
        except Exception as e:
            print(f"[warn] Failed to write cache {path}: {e}")

    # Core eval (single call per seed) -----------------------------------
    def _call_with_retries(self, fn, *args, **kwargs) -> Tuple[Set[str], Dict, int, Optional[str]]:
        max_tries = max(1, int(self.args.retries) + 1)
        delay = 0.6
        for attempt in range(1, max_tries + 1):
            t0 = time.time()
            try:
                acc, ver = fn(*args, **kwargs)
                ms = int((time.time() - t0) * 1000)
                return acc, ver, ms, None
            except Exception as e:
                ms = int((time.time() - t0) * 1000)
                if attempt >= max_tries:
                    return set(), {"phase": "?", "schema_used": "?", "unknown_terms": [], "duplicates": [], "relations": {}}, ms, str(e)
                # Exponential backoff with jitter
                sleep_s = delay * (2 ** (attempt - 1)) * (1.0 + 0.25 * random.random())
                time.sleep(min(sleep_s, 5.0))
        return set(), {"phase": "?", "schema_used": "?", "unknown_terms": [], "duplicates": [], "relations": {}}, 0, "unknown"

    def eval_seed_batch(self, seed: str, candidates: List[str], anchors: Optional[List[str]]) -> Tuple[Set[str], Dict, int, Optional[str]]:
        return self._call_with_retries(self.judge.judge_batch, seed, candidates, anchors)

    # Expand batch result into per-candidate DecisionRecords
    def expand_records(self, seed: str, candidates: List[str], accepted: Set[str], ver: Dict, latency_ms: int, err: Optional[str]) -> Tuple[Set[str], List[DecisionRecord], List[str]]:
        records: List[DecisionRecord] = []
        aligned: List[str] = []
        prompt_type_base = ver.get("phase", "batch")
        schema_used = ver.get("schema_used", "batch")
        unknown_terms = ver.get("unknown_terms", [])
        duplicates = ver.get("duplicates", [])
        schema_ok = ver.get("schema_ok", True)
        relations_map: Dict[str, str] = ver.get("relations", {}) if isinstance(ver.get("relations", {}), dict) else {}

        # Optionally force-accept equality, and mark those as 'shortcut'
        accepted_eff = set(accepted)
        equality_forced: Set[str] = set()
        if self.args.auto_accept_if_equal:
            for cand in candidates:
                if cand == seed and cand not in accepted_eff:
                    accepted_eff.add(cand)
                    equality_forced.add(cand)

        for cand in candidates:
            is_forced = cand in equality_forced
            if err is not None:
                rec = DecisionRecord(
                    seed=seed, candidate=cand, accepted=False, decision="error",
                    prompt_type=prompt_type_base, schema_used=schema_used,
                    unknown_terms=unknown_terms, duplicates=duplicates,
                    attempts=max(1, int(self.args.retries) + 1), latency_ms=latency_ms, error=str(err),
                    relation=(relations_map.get(cand) if not is_forced else ("synonym" if self.args.relation_mode else None)),
                )
            else:
                ok = cand in accepted_eff
                if ok:
                    decision = "accept"
                elif (not schema_ok) or unknown_terms or duplicates:
                    decision = "unknown_mismatch"
                else:
                    decision = "reject"
                rec = DecisionRecord(
                    seed=seed, candidate=cand, accepted=ok, decision=decision,
                    prompt_type=("shortcut" if is_forced else prompt_type_base), schema_used=schema_used,
                    unknown_terms=unknown_terms, duplicates=duplicates,
                    attempts=1, latency_ms=latency_ms, error=None,
                    relation=(relations_map.get(cand) if not is_forced else ("synonym" if self.args.relation_mode else None)),
                )
            records.append(rec)
            aligned.append(cand if rec.accepted else "")
            if self.args.cache_path:
                self._append_cache(self.args.cache_path, rec)

        # Return the equality-augmented set so seed-level outputs match single-term behavior
        return accepted_eff, records, aligned

    # Per-seed sequential flow (one LLM call)
    def process_seed(self, seed: str, candidates: List[str]) -> Tuple[Set[str], List[DecisionRecord], List[str]]:
        anchors_ctx: Optional[List[str]] = self.global_anchors if self.args.use_anchors and self.global_anchors else None
        accepted, ver, ms, err = self.eval_seed_batch(seed, candidates, anchors_ctx)
        return self.expand_records(seed, candidates, accepted, ver, ms, err)

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
    logger = logging.getLogger("batch_eval")
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

    logger.info("# Batch LLM evaluation summary (LLM-only WordNet-like relations)")
    logger.info(f"Total evals: {total} | accept={acc} reject={rej} unknown_mismatch={unk} error={err}")
    lat_ok = [r.latency_ms for r in all_records if r.decision in {"accept", "reject", "unknown_mismatch"}]
    if lat_ok:
        logger.info(
            f"Latency (ms) median={int(percentile(lat_ok,50))} p90={int(percentile(lat_ok,90))} max={max(lat_ok)}"
        )
    logger.info("")

    # Per-seed acceptance rates (+ quick relation overview if present)
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
        logger.info(
            f"{seed:<28s} | total={len(seed_recs):4d} accept={acc_s:4d} reject={rej_s:4d} unknown={unk_s:3d} error={err_s:3d}"
        )
    logger.info("")

    # Verification issues
    logger.info("## verification issues (model must return a subset of provided candidates)")
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
# Main (sequential across seeds, one call per seed)
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Batch LLM judgments for seed expansions (single call per seed; LLM-only WordNet-like relations).")
    ap.add_argument("--expansions", required=True, help="Path to expansions JSON {seed: [terms]}")
    ap.add_argument("--outdir", required=False, help="Directory for outputs (ignored in debug mode)")

    # LLM
    ap.add_argument("--model", type=str, default="llama3.3:latest", help="Ollama model name")
    ap.add_argument("--url", type=str, default="http://localhost:11434/api/chat", help="Ollama chat endpoint")
    ap.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    ap.add_argument("--tokens", type=int, default=2048, help="LLM context tokens")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    ap.add_argument("--global_context", type=str, default=GLOBAL_CONTEXT_DEFAULT,
        help="Short domain context sentence(s) injected into prompts")

    # Execution controls
    ap.add_argument("--retries", type=int, default=2, help="Retry count on request/parse errors (exponential backoff)")
    ap.add_argument("--sort", action="store_true", help="Sort final lists alphabetically where applicable")
    ap.add_argument("--auto_accept_if_equal", action="store_true", help="Auto-accept when candidate == seed (post-parse)")

    # Filtering / limits
    ap.add_argument("--seed_filter", type=str, nargs="*", default=None, help="Only process these seeds (exact match)")
    ap.add_argument("--limit_per_seed", type=int, default=0, help="Limit number of candidates per seed (0=all)")

    # Anchors
    ap.add_argument("--use_anchors", action="store_true",
                    help="If set, the batch prompt uses the FULL seed vocabulary (all seeds in the JSON) as anchors/context.")

    # Relations + closure
    ap.add_argument("--relation_mode", action="store_true",
                    help="If set, the model may also output a 'relations' map with a single relation label per accepted candidate.")
    ap.add_argument("--closure_iters", type=int, default=0,
                    help="How many iterative closure rounds the model should apply within the candidate list (0..N). Default 0 for parity with single-term.")

    # Caching
    ap.add_argument("--cache_path", type=str, default=None, help="Optional JSONL cache file for per-pair decisions (append-only)")

    # Debug
    ap.add_argument("--debug_seed", type=str, default=None, help="If set, evaluate only this seed (one LLM call) and optionally show one candidate")
    ap.add_argument("--debug_term", type=str, default=None, help="Used with --debug_seed to print the specific (seed,candidate) decision line")

    args = ap.parse_args()

    expansions = load_json(args.expansions)
    if not isinstance(expansions, dict):
        raise ValueError("Expansions JSON must be an object mapping seeds to lists of terms.")

    # Prepare global anchors (full seed vocabulary) if requested
    global_seed_vocab: List[str] = sorted([str(s) for s, v in expansions.items() if isinstance(v, list) and v]) if args.use_anchors else []

    # --- DEBUG MODE: one seed call ---
    if args.debug_seed:
        seed = args.debug_seed
        cand_list_full = expansions.get(seed)
        if not isinstance(cand_list_full, list):
            print(f"[warn] Seed '{seed}' not found or expansions not a list.")
            return
        candidates = [str(c) for c in cand_list_full]
        if args.limit_per_seed and args.limit_per_seed > 0:
            candidates = candidates[: args.limit_per_seed]

        runner = BatchRunner(args, global_anchors=global_seed_vocab)
        accepted, ver, ms, err = runner.eval_seed_batch(seed, candidates, global_seed_vocab if args.use_anchors else None)
        acc_set, records, aligned = runner.expand_records(seed, candidates, accepted, ver, ms, err)

        if args.debug_term:
            rec = next((r for r in records if r.candidate == args.debug_term), None)
            if rec is None:
                print(f"[warn] Candidate '{args.debug_term}' not in list for seed '{seed}'.")
            else:
                print("=== DEBUG SINGLE-PAIR (from batch) ===")
                print(json.dumps(asdict(rec), indent=2, ensure_ascii=False))
        else:
            print("=== DEBUG BATCH ===")
            print(json.dumps({
                "seed": seed,
                "accepted": sorted(list(acc_set)),
                "schema_ok": ver.get("schema_ok", False),
                "unknown_terms": ver.get("unknown_terms", []),
                "duplicates": ver.get("duplicates", []),
                "relations": ver.get("relations", {}),
                "latency_ms": ms,
                "error": err,
            }, indent=2, ensure_ascii=False))
        return

    # --- NORMAL MODE (sequential across seeds) ---
    if not args.outdir:
        raise ValueError("--outdir is required unless --debug_seed is set")

    outdir = Path(args.outdir).expanduser().resolve()
    timestamp = datetime.now().strftime("%m_%d_%H_%M")
    eval_dir = outdir / f"batch_eval_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_path = eval_dir / "batch_eval.log"
    ndjson_pairs_path = eval_dir / "decisions.ndjson"
    ndjson_seeds_path = eval_dir / "seeds_accepted.ndjson"
    accepted_by_seed_path = eval_dir / "accepted_by_seed.json"
    filtered_expansions_path = eval_dir / "filtered_expansions.json"  # same shape as input
    accepted_all_flat_path = eval_dir / "accepted_all_flat.json"
    accepted_aligned_path = eval_dir / "accepted_aligned_by_seed.json"
    summary_path = eval_dir / "summary.json"

    logger = setup_logger(log_path)
    runner = BatchRunner(args, global_anchors=global_seed_vocab)

    # Filter seeds if requested
    seeds = list(expansions.keys())
    if args.seed_filter:
        seed_set = set(args.seed_filter)
        seeds = [s for s in seeds if s in seed_set]

    # Build job list (sequential)
    jobs: List[Tuple[str, List[str]]] = []
    for seed in seeds:
        cands = expansions.get(seed, [])
        if not isinstance(cands, list):
            logger.info(f"[warn] Seed '{seed}' expansions not a list; treating as empty.")
            cands = []
        else:
            cands = [str(c) for c in cands]  # coerce to strings
        if args.limit_per_seed and args.limit_per_seed > 0:
            cands = cands[: args.limit_per_seed]
        jobs.append((seed, cands))

    all_records: List[DecisionRecord] = []
    accepted_by_seed: Dict[str, List[str]] = {}
    aligned_by_seed: Dict[str, List[str]] = {}
    per_seed_rows: List[dict] = []

    # Process seeds sequentially (no concurrency)
    logger.info("# Concurrency disabled: server serializes per-model requests; processing seeds sequentially (one call per seed).")
    for seed, cand_list in jobs:
        acc, recs, aligned = runner.process_seed(seed, cand_list)

        # Subset check + logging: make sure accepted ⊆ original expansions
        original_list = expansions.get(seed, [])
        original_set = set(original_list) if isinstance(original_list, list) else set()
        violations = sorted([t for t in acc if t not in original_set])
        if violations:
            # Drop violators from persisted outputs, but log them
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

    # Build flat list of all accepted terms across seeds in true encounter order
    accepted_all_flat: List[str] = []
    for seed, _cands in jobs:
        aligned = aligned_by_seed.get(seed, [])
        for term in aligned:
            if term:
                accepted_all_flat.append(term)
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
        "tokens": args.tokens,
        "timeout": args.timeout,
        "use_anchors": bool(args.use_anchors),
        "retries": args.retries,
        "sorted": bool(args.sort),
        "relation_mode": bool(args.relation_mode),
        "closure_iters": int(args.closure_iters),
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
    logger.info(f"# UseAnchors={bool(args.use_anchors)} Retries={args.retries} Sort={bool(args.sort)} RelationMode={bool(args.relation_mode)} ClosureIters={int(args.closure_iters)}")
    logger.info("")
    log_summary(logger, all_records, accepted_by_seed)

if __name__ == "__main__":
    main()
