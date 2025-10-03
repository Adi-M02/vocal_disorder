#!/usr/bin/env python3
"""
Single-term evaluator with CORPUS CONTEXT (STRICT one-candidate schema)
Parallelized across multiple local Ollama instances
======================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import sys

sys.path.append('../vocal_disorder')

import requests

DEFAULT_OLLAMA_ENDPOINTS = [
    "http://localhost:11434/api/chat",
    "http://localhost:11435/api/chat",
    "http://localhost:11436/api/chat",
    "http://localhost:11437/api/chat",
]

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

class LlmDeciderWithContext:
    def __init__(
        self,
        model: str,
        url: str = "http://localhost:11436/api/chat",
        temperature: float = 0.0,
        tokens: int = 8192,
        timeout: int = 60,
        session: Optional[requests.Session] = None,
    ):
        self.url = url
        self.timeout = timeout
        self.session = session or requests.Session()
        self.base_payload = {
            "model": model,
            "options": {"temperature": float(temperature), "num_ctx": int(tokens)},
            "stream": False,
        }
        self.system_prompt = (
            "You decide whether a short term or underscore-separated MWE CANDIDATE should be ACCEPTED "
            "as essentially the same concept as a given SEED within Reddit discussions about R-CPD "
            "(Retrograde Cricopharyngeus Dysfunction) and closely related care.\n"
            "\n"
            "Acceptance should be generous but sane: ACCEPT when the CANDIDATE is a synonym, near-synonym, "
            "common rephrasing, paraphrase, abbreviation/acronym, spelling/hyphenation/underscore variant, "
            "singular/plural or morphological/part-of-speech variant (verb↔noun), or a very common lay vs. clinical "
            "name that users would treat interchangeably for the same thing in this domain. Also ACCEPT when the "
            "CANDIDATE reliably refers to the same action, procedure, symptom, body structure, test, or therapy "
            "as the SEED in context.\n"
            "\n"
            "Ground your decision in the provided **corpus usage windows** for the SEED and CANDIDATE and the "
            "ANCHOR_TERMS list (a background domain lexicon).\n"
            "- If the CANDIDATE’s usage windows could replace the SEED in those sentences without changing meaning, "
            "lean ACCEPT.\n"
            "- If multiple CANDIDATE windows show the same role/topic as the SEED (e.g., same procedure, same body region, "
            "same purpose), lean ACCEPT.\n"
            "- If windows suggest different entities or contrasting procedures (e.g., dilation vs. botox) or clearly unrelated "
            "meanings, REJECT.\n"
            "- Anchors are weak evidence: overlap with anchors (e.g., cricopharyngeus/UES/botox/ENT/manometry) strengthens ACCEPT "
            "if usage also aligns.\n"
            "- When uncertain but windows are plausibly interchangeable, prefer ACCEPT.\n"
            "\n"
            "STRICT OUTPUT:\n"
            "Return JSON only with keys {seed, decisions}. For 'decisions' return either [] (reject) or [<exact candidate>] (accept). "
            "Do NOT alter or normalize any strings, and do NOT include explanations."
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
    def _fmt_windows(windows: List[List[str]]) -> str:
        """
        DO NOT modify, truncate, or cap snippets.
        Flatten per-window so the LLM sees each returned snippet exactly as provided by the query function.
        """
        if not windows:
            return "(none)"
        flat_snips: List[str] = []
        for winlist in windows:           # per document
            for snippet in winlist:       # per window occurrence
                flat_snips.append(f"- {snippet}")
        return " ".join(flat_snips) if flat_snips else "(none)"

    def _build_user_prompt(self,
                           seed: str,
                           candidate: str,
                           anchors: List[str],
                           seed_windows: List[List[str]],
                           cand_windows: List[List[str]],
                           max_anchors: int = 500) -> str:
        anc_str = " ".join(f"- {a}" for a in anchors[:max_anchors]) if anchors else "(none)"
        seed_ctx = self._fmt_windows(seed_windows)
        cand_ctx = self._fmt_windows(cand_windows)

        guidance = (
            "DECISION GOAL:\n"
            "Accept if the candidate is used in the corpus like the seed (same underlying concept) — including synonyms, "
            "near-synonyms, paraphrases/rephrasings, abbreviations/acronyms, spelling and underscore/hyphen variants, "
            "singular/plural or other morphological variants, or standard lay vs. clinical wordings. "
            "If usages point to clearly different or contrasting things, reject.\n"
            "\n"
            "HOW TO USE THE EVIDENCE:\n"
            "1) Compare SEED vs CANDIDATE windows. If they look interchangeable (same role/procedure/body region/symptom/test/therapy), lean ACCEPT.\n"
            "2) Look for repeated alignment across multiple CANDIDATE windows; that strengthens ACCEPT.\n"
            "3) Use ANCHOR_TERMS as soft context: overlaps with anchors that fit the same topic (e.g., UES, cricopharyngeus, botox, ENT, manometry) "
            "support ACCEPT when windows also align.\n"
            "4) If CANDIDATE frequently appears contrasted with the SEED (e.g., 'X instead of Y', 'X vs Y'), or in unrelated topics, REJECT."
        )

        base = (
            f"SEED: {seed}\n"
            f"CANDIDATE (evaluate only this exact string; accept ⇒ return it, reject ⇒ return []):\n"
            f"- {candidate}\n\n"
            f"{guidance}\n\n"
            f"SEED_CORPUS_WINDOWS (usage contexts): {seed_ctx}\n"
            f"CANDIDATE_CORPUS_WINDOWS (usage contexts): {cand_ctx}\n"
            f"ANCHOR_TERMS (background lexicon; soft evidence): {anc_str}\n"
        )

        return base + 'Respond ONLY with JSON {"seed": <seed>, "decisions": [] or [<exact candidate>]}.'


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
            if isinstance(content, str):
                return json.loads(content)
            if isinstance(content, dict):
                return content
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
        # Debug print of user message (comment out if too verbose)
        # print(payload["messages"][1]["content"])
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

@dataclass
class DecisionRecord:
    seed: str
    candidate: str
    accepted: bool
    decision: str
    schema_used: str = "unknown"
    unknown_terms: List[str] = field(default_factory=list)
    duplicates: List[str] = field(default_factory=list)
    attempts: int = 1
    latency_ms: int = 0
    error: Optional[str] = None
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

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

class SingleTermRunner:
    def __init__(self, args: argparse.Namespace, anchors: Optional[List[str]] = None, override_url: Optional[str] = None):
        self.args = args
        self.session = requests.Session()
        self.judge = LlmDeciderWithContext(
            model=args.model,
            url=(override_url or args.url),
            temperature=args.temperature,
            tokens=args.tokens,
            timeout=args.timeout,
            session=self.session,
        )
        self.anchors: List[str] = list(anchors or [])

        # Lazy windows: load corpus once; compute per-term on demand
        from testing.adding_context.context_api import load_cached_corpus, query_base_windows
        self._ctx_df = load_cached_corpus(args.cache_path, format=args.context_format)
        self._query_base_windows = query_base_windows
        self._window_cache: Dict[str, List[List[str]]] = {}

    def _windows_for(self, term: str) -> List[List[str]]:
        wins = self._window_cache.get(term)
        if wins is None:
            wins = self._query_base_windows(self._ctx_df, term, self.args.k, self.args.window)
            self._window_cache[term] = wins if isinstance(wins, list) else []
        return wins

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
        logger.info(f"Latency (ms) median={int(percentile(lat_ok,50))} p90={int(percentile(lat_ok,90))} max={max(lat_ok)}")
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

def _normalize_expansions(expansions: dict) -> dict:
    """Ensure every seed maps to a list[str]."""
    out = {}
    for seed, cands in expansions.items():
        if isinstance(cands, list):
            out[str(seed)] = [str(c) for c in cands]
        else:
            out[str(seed)] = []
    return out

def _split_round_robin(items: List[str], n: int) -> List[List[str]]:
    chunks = [[] for _ in range(n)]
    for i, it in enumerate(items):
        chunks[i % n].append(it)
    return chunks

def _process_seed_subset(
    url: str,
    args: argparse.Namespace,
    anchors: List[str],
    expansions_norm: Dict[str, List[str]],
    seeds_subset: List[str],
) -> Tuple[List[DecisionRecord], Dict[str, List[str]], Dict[str, List[str]], List[dict]]:
    """
    Worker: process a subset of seeds against a specific Ollama endpoint.
    Returns (records, accepted_by_seed, aligned_by_seed, per_seed_rows).
    """
    runner = SingleTermRunner(args, anchors=anchors, override_url=url)
    all_records: List[DecisionRecord] = []
    accepted_by_seed: Dict[str, List[str]] = {}
    aligned_by_seed: Dict[str, List[str]] = {}
    per_seed_rows: List[dict] = []

    for seed in seeds_subset:
        cand_list = expansions_norm.get(seed, [])
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
        original_set = set(expansions_norm.get(seed, []))
        violations = sorted([t for t in acc_set if t not in original_set])
        if violations:
            acc_set = {t for t in acc_set if t in original_set}
        acc_list = sorted(acc_set) if args.sort else list(acc_set)
        accepted_by_seed[seed] = acc_list
        aligned_by_seed[seed] = aligned
        per_seed_rows.append({"seed": seed, "accepted": acc_list, "checked_subset": len(violations) == 0, "violations": violations})
        all_records.extend(recs)

    return all_records, accepted_by_seed, aligned_by_seed, per_seed_rows

def main():
    ap = argparse.ArgumentParser(description="Single-term LLM judgments with corpus context (STRICT schema, R-CPD aware).")
    ap.add_argument("--expansions", default='testing/ngram_evals_test_no_digits/4/topk_25_min_cos_0.4_cbow.json', help="Path to expansions JSON {seed: [terms]}")
    ap.add_argument("--outdir", required=False, default=None, help="Directory for outputs (required unless in debug mode)")
    ap.add_argument("--model", type=str, default="llama3.3:latest", help="Ollama model name")
    ap.add_argument("--url", type=str, default="http://localhost:11436/api/chat", help="Default Ollama chat endpoint (used in debug mode or single-instance)")
    ap.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    ap.add_argument("--tokens", type=int, default=8192, help="LLM context tokens")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    ap.add_argument("--retries", type=int, default=2, help="Retry count on errors (exponential backoff)")
    ap.add_argument("--sort", action="store_true", help="Sort final accepted lists alphabetically where applicable")
    ap.add_argument("--use_anchors", action="store_true", default=True, help="Include anchors = expansions' keys that have non-empty lists")
    ap.add_argument("--cache_path", type=str, default="testing/adding_context/cache/ngram_df_baseTokens_and_ngramText.parquet", help="Path to cached corpus (e.g., parquet)")
    ap.add_argument("--context_format", type=str, default="parquet", help="Cache format (e.g., parquet)")
    ap.add_argument("--k", type=int, default=3, help="Number of windows per term to include")
    ap.add_argument("--window", type=int, default=50, help="Half-window size (tokens to left/right) or implementation-defined")
    ap.add_argument("--debug_seed", type=str, default=None, help="If set, only evaluate this seed")
    ap.add_argument("--debug_candidates", type=str, nargs="+", default=None, help="Space- or comma-separated candidates")
    ap.add_argument("--instances", type=int, default=4, help="Number of Ollama instances to use in parallel (up to 4; ports 11434..11437)")
    args = ap.parse_args()

    debug_mode = (args.debug_seed is not None and args.debug_candidates is not None)

    expansions = load_json(args.expansions)
    if not isinstance(expansions, dict):
        raise ValueError("Expansions JSON must be an object mapping seeds to lists of terms.")
    expansions_norm = _normalize_expansions(expansions)

    anchors: List[str] = sorted([str(s) for s, v in expansions_norm.items() if isinstance(v, list) and v]) if args.use_anchors else []

    # Debug mode: original sequential behavior, single endpoint (args.url)
    if debug_mode:
        runner = SingleTermRunner(args, anchors=anchors, override_url=args.url)
        dbg: List[str] = []
        for item in args.debug_candidates:
            dbg += [t.strip() for t in item.split(",") if t.strip()]
        jobs = [(args.debug_seed, [str(c) for c in dbg])]
        for seed, cand_list in jobs:
            for cand in cand_list:
                rec = runner.eval_pair(seed, cand)
                print(cand if rec.accepted else "")
        return

    if not args.outdir:
        raise SystemExit("--outdir is required unless running in debug mode.")

    # Parallel run across N instances
    instances = max(1, min(int(args.instances), len(DEFAULT_OLLAMA_ENDPOINTS)))
    endpoints = DEFAULT_OLLAMA_ENDPOINTS[:instances]

    seeds = list(expansions_norm.keys())
    seed_chunks = _split_round_robin(seeds, instances)

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

    # Launch workers
    futures = []
    with ThreadPoolExecutor(max_workers=instances) as ex:
        for idx, subset in enumerate(seed_chunks):
            if not subset:
                continue
            url = endpoints[idx]
            futures.append(
                ex.submit(_process_seed_subset, url, args, anchors, expansions_norm, subset)
            )

        # Collect results
        all_records: List[DecisionRecord] = []
        accepted_by_seed: Dict[str, List[str]] = {}
        aligned_by_seed: Dict[str, List[str]] = {}
        per_seed_rows: List[dict] = []

        for fut in as_completed(futures):
            recs, acc_map, aligned_map, rows = fut.result()
            all_records.extend(recs)
            accepted_by_seed.update(acc_map)
            aligned_by_seed.update(aligned_map)
            per_seed_rows.extend(rows)

    # Write outputs (same as before)
    write_ndjson(ndjson_pairs_path, all_records)
    write_seed_ndjson(ndjson_seeds_path, per_seed_rows)
    save_json(accepted_by_seed_path, accepted_by_seed)
    save_json(filtered_expansions_path, accepted_by_seed)
    save_json(accepted_aligned_path, aligned_by_seed)

    accepted_all_flat: List[str] = []
    for seed in expansions_norm.keys():
        aligned = aligned_by_seed.get(seed, [])
        for t in aligned:
            if t:
                accepted_all_flat.append(t)
    save_json(accepted_all_flat_path, accepted_all_flat)

    seeds_with_errors = len({r.seed for r in all_records if r.decision == "error"})
    seeds_all_empty = sum(1 for s in expansions_norm.keys() if not accepted_by_seed.get(s))
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
        "default_url": args.url,
        "instances": instances,
        "endpoints_used": endpoints,
        "temperature": args.temperature,
        "tokens": args.tokens,
        "timeout": args.timeout,
        "retries": args.retries,
        "sorted": bool(args.sort),
        "use_anchors": bool(args.use_anchors),
        "anchors_count": len(anchors),
        "cache_path": args.cache_path,
        "k": args.k,
        "window": args.window,
    }
    save_json(summary_path, summary)

    logger.info(f"# Parallel instances: {instances} -> {', '.join(endpoints)}")
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
