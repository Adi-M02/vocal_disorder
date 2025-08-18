#!/usr/bin/env python3
"""
Augment seed terms using an LLM (via Ollama) to judge semantic relatedness
between each seed and its expansion candidates.

Response schema (LLM):
  {
    "seed": "<seed>",
    "decisions": ["<accepted_term_1>", "<accepted_term_2>", ...]   # exact, unchanged strings from input
  }

Verification:
  - Ensures all returned terms are EXACT members of the input candidate list.
  - Unknown/altered terms are ignored for acceptance and logged (with duplicates).

Usage (normal mode):
  python augment_seeds_with_llm.py \
      --expansions path/to/expansions.json \
      --outdir path/to/output_dir \
      --model llama3.3:latest \
      [--batch_size 80] [--max_per_seed 0] [--sort]

Usage (debug mode; prints only one seed, writes nothing):
  python augment_seeds_with_llm.py \
      --expansions path/to/expansions.json \
      --model llama3.3:latest \
      --debug_seed get_stuck
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, Iterable, List, Set, Tuple

import requests

# -----------------------------
# Normalization helpers
# -----------------------------
def norm(tok: str) -> str:
    return tok.strip().lower()

def join_uniq_sorted(iterable: Iterable[str]) -> str:
    return ", ".join(sorted(set(iterable)))

# -----------------------------
# Verification helper (strings-only)
# -----------------------------
def verify_unchanged(candidates: List[str], returned_terms: List[str]) -> Dict[str, List[str]]:
    """
    Compare model-returned terms to the original candidates (exact string match).
    Returns dict with 'unknown_terms' and 'duplicates'.
    """
    cand_set = set(candidates)  # exact forms
    decided_terms = [t for t in returned_terms if isinstance(t, str)]
    unknown = [t for t in decided_terms if t not in cand_set]
    dupes = [t for t, c in Counter(decided_terms).items() if c > 1]
    return {"unknown_terms": sorted(set(unknown)), "duplicates": sorted(dupes)}

# -----------------------------
# Ollama LLM client
# -----------------------------
class LlmSimilarityDecider:
    """
    Calls an Ollama chat model expecting:
      {"seed": "<seed>", "decisions": ["termA","termB",...]}
    """

    def __init__(
        self,
        model: str,
        url: str = "http://localhost:11434/api/chat",
        temperature: float = 0.0,
        timeout: int = 60,
        batch_size: int = 80,
    ):
        self.url = url
        self.timeout = timeout
        self.batch_size = max(1, int(batch_size))
        self.headers = {"Content-Type": "application/json"}

        # Strict schema: decisions are strings only
        self.format_schema = {
            "type": "object",
            "properties": {
                "seed": {"type": "string"},
                "decisions": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["seed", "decisions"]
        }

        self.base_payload = {
            "model": model,
            "options": {"temperature": float(temperature)},
            "stream": False,
            "format": self.format_schema
        }

        # System prompt: underscore MWEs, semantic head/variants, decisions are exact, unchanged strings
        self.system_message = (
            "You are a semantic similarity decision maker for short terms and underscore-separated MWEs.\n"
            "Given a SEED and a list of CANDIDATES, return the subset of candidates that are semantically similar to "
            "the seed in a practical vocabulary-expansion sense (near-synonyms, common paraphrases, morphological "
            "variants, closely related alternatives, used in a similar context or MWEs that include the seed's semantic head or a close variant).\n\n"
            "CRITICAL:\n"
            " • Consider candidates EXACTLY as given; do NOT normalize, spell-correct, or modify text.\n"
            " • Return STRICT JSON ONLY with keys: seed, decisions.\n"
            " • decisions MUST be a list of the accepted candidate STRINGS, UNCHANGED from input.\n"
        )

    def _build_user_prompt(self, seed: str, candidates: List[str]) -> str:
        cand_str = "\n".join(f"- {c}" for c in candidates)
        return (
            f"SEED: {seed}\n"
            f"CANDIDATES (one per line; return accepted ones unchanged):\n{cand_str}\n\n"
            "Respond ONLY with JSON per the enforced schema."
        )

    def _post(self, payload: dict) -> dict:
        resp = requests.post(self.url, headers=self.headers, json=payload, timeout=self.timeout)
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
        """
        Coerce the response into a list of strings (accepted terms).
        Returns (terms, schema_used)
        """
        raw = out.get("decisions", []) if isinstance(out, dict) else []
        schema_used = "string"

        if not isinstance(raw, list):
            return [], "empty"

        # If the model still returns objects, try to extract 'term' strings (best-effort)
        terms: List[str] = []
        if raw and isinstance(raw[0], dict):
            schema_used = "object->string"
            for item in raw:
                t = item.get("term") if isinstance(item, dict) else None
                if isinstance(t, str):
                    terms.append(t)
        else:
            schema_used = "string"
            terms = [t for t in raw if isinstance(t, str)]

        return terms, schema_used

    def judge_batch(self, seed: str, candidates: List[str]) -> Tuple[List[str], Dict]:
        """
        Judge a single batch; returns (accepted_terms, verification_info).
        """
        payload = dict(self.base_payload)
        payload["messages"] = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self._build_user_prompt(seed, candidates)},
        ]
        out = self._post(payload)
        terms, schema_used = self._coerce_terms(out)

        # Verification: returned terms must be exact candidates
        ver = verify_unchanged(candidates, terms)
        ver["schema_used"] = schema_used
        return terms, ver

    def judge_all(self, seed: str, candidates: List[str]) -> Tuple[List[str], Dict]:
        """
        Batch across all candidates. Returns:
          (all_accepted_terms, merged_verification_info)
        """
        if not candidates:
            return [], {"schema_used": "empty", "unknown_terms": [], "duplicates": []}

        all_terms: List[str] = []
        agg_unknown: Set[str] = set()
        agg_dupes: Set[str] = set()
        schema_seen: Set[str] = set()

        N = len(candidates)
        for i in range(0, N, self.batch_size):
            chunk = candidates[i:i + self.batch_size]
            terms, ver = self.judge_batch(seed, chunk)
            all_terms.extend(terms)
            agg_unknown |= set(ver.get("unknown_terms", []))
            agg_dupes   |= set(ver.get("duplicates", []))
            schema_seen.add(ver.get("schema_used", "unknown"))

        merged_ver = {
            "schema_used": "/".join(sorted(schema_seen)) if schema_seen else "empty",
            "unknown_terms": sorted(agg_unknown),
            "duplicates": sorted(agg_dupes),
        }
        return all_terms, merged_ver

# -----------------------------
# Selection logic per seed (LLM-based, strings-only)
# -----------------------------
def select_with_llm_for_seed(
    judge: LlmSimilarityDecider,
    seed: str,
    cand_list: List[str],
    *,
    max_per_seed: int = 0
) -> Tuple[Set[str], Dict[str, str], Dict]:
    """
    Returns (accepted, reasons_map, dbg) for a single seed using the LLM.
    Acceptance = intersection of model-accepted strings and the original candidates (exact match).
    """
    seed_n = norm(seed)
    # IMPORTANT: send/verify EXACT forms to the model; no lowercasing here
    cands = [t for t in cand_list if isinstance(t, str) and t.strip()]
    cand_set = set(cands)

    accepted_terms, ver = judge.judge_all(seed_n, cands)

    # Keep only exact matches
    accepted = set(t for t in accepted_terms if t in cand_set)

    if max_per_seed > 0 and len(accepted) > max_per_seed:
        accepted = set(sorted(list(accepted))[:max_per_seed])

    # Simple reasons map
    reasons = {t: "llm:accepted" for t in accepted}

    dbg = {
        "accepted": sorted(accepted),
        "verification": ver,  # includes schema_used, unknown_terms, duplicates
    }
    return accepted, reasons, dbg

# -----------------------------
# I/O
# -----------------------------
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# -----------------------------
# Logging helpers
# -----------------------------
def setup_file_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("augment_llm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    return logger

def log_per_seed_lines(logger: logging.Logger, added_by_seed: Dict[str, List[Tuple[str, str]]]):
    if not added_by_seed:
        logger.info("No additions recorded.")
        return
    maxw = max(len(s) for s in added_by_seed.keys()) if added_by_seed else 12
    maxw = max(12, min(maxw, 28))
    for seed in sorted(added_by_seed.keys()):
        pairs = added_by_seed[seed]
        if not pairs:
            continue
        right = ", ".join(f"{t}({r})" for t, r in sorted(pairs, key=lambda x: (x[1], x[0])))
        logger.info(f"{seed:<{maxw}s} -> {right}")

def log_verification_issues(logger: logging.Logger, verify_map: Dict[str, Dict[str, List[str]]]):
    any_issues = any(v.get("unknown_terms") or v.get("duplicates") for v in verify_map.values())
    logger.info("## verification issues (terms must be returned unchanged)")
    if not any_issues:
        logger.info("(none)")
        return
    for seed in sorted(verify_map.keys()):
        v = verify_map[seed]
        sch = v.get("schema_used", "unknown")
        unk = v.get("unknown_terms", [])
        dup = v.get("duplicates", [])
        parts = [f"schema={sch}"]
        if unk: parts.append(f"unknown={','.join(unk[:20])}{'…' if len(unk)>20 else ''}")
        if dup: parts.append(f"duplicates={','.join(dup)}")
        logger.info(f"{seed} :: " + " | ".join(parts))

# -----------------------------
# DEBUG printer
# -----------------------------
def print_debug(seed: str, cand_list: List[str], args):
    judge = LlmSimilarityDecider(
        model=args.model,
        url=args.url,
        temperature=args.temperature,
        timeout=args.timeout,
        batch_size=args.batch_size,
    )
    accepted, reasons, dbg = select_with_llm_for_seed(
        judge,
        seed,
        cand_list,
        max_per_seed=args.max_per_seed
    )

    print("=== DEBUG MODE (LLM) ===")
    print(f"Seed               : {seed}")
    print(f"Model / URL        : {args.model} / {args.url}")
    print(f"Batch size         : {args.batch_size}")
    print("")
    print(f"Accepted [{len(accepted)}]: {join_uniq_sorted(accepted) or '(none)'}")
    print("")
    print("Verification:")
    v = dbg["verification"]
    print(f"  schema_used  : {v.get('schema_used','unknown')}")
    print(f"  unknown_terms: {', '.join(v.get('unknown_terms', [])) or '(none)'}")
    print(f"  duplicates   : {', '.join(v.get('duplicates', [])) or '(none)'}")
    print("")
    print("Reasons (first 25):")
    for i, (t, r) in enumerate(sorted(reasons.items())):
        if i >= 25: break
        print(f"  {t:<30s} -> {r}")
    print("=========================")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Augment seed terms using an Ollama LLM to judge semantic relatedness.")
    ap.add_argument("--expansions", required=True, help="Path to expansions JSON {seed: [terms]}")
    ap.add_argument("--outdir", help="Directory for outputs (ignored in --debug_seed mode)")
    ap.add_argument("--model", type=str, default="llama3.3:latest", help="Ollama model name")
    ap.add_argument("--url", type=str, default="http://localhost:11434/api/chat", help="Ollama chat endpoint")
    ap.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    ap.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds")
    ap.add_argument("--batch_size", type=int, default=1, help="Candidates per LLM call")
    ap.add_argument("--max_per_seed", type=int, default=0, help="Cap accepted terms per seed (0=off)")
    ap.add_argument("--sort", action="store_true", help="Sort final seed list alphabetically")
    ap.add_argument("--debug_seed", type=str, default=None,
                    help="If set, only process this seed and print results to console (no files written).")
    args = ap.parse_args()

    expansions = load_json(args.expansions)
    if not isinstance(expansions, dict):
        raise ValueError("Expansions JSON must be an object mapping seeds to lists of terms.")

    # --- DEBUG MODE ---
    if args.debug_seed:
        wanted = norm(args.debug_seed)
        key = None
        for k in expansions.keys():
            if norm(k) == wanted:
                key = k
                break
        if key is None:
            print(f"[warn] Seed '{args.debug_seed}' not found in expansions.")
            return
        cand_list = expansions.get(key, [])
        if not isinstance(cand_list, list):
            print(f"[warn] Expansions for seed '{key}' are not a list.")
            return
        print_debug(key, cand_list, args)
        return

    # --- NORMAL MODE ---
    if not args.outdir:
        raise ValueError("--outdir is required unless --debug_seed is set")

    outdir = Path(args.outdir).expanduser().resolve()
    timestamp = datetime.now().strftime("%m_%d_%H_%M")
    eval_dir = outdir / f"added_eval_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    json_out_path = eval_dir / "new_eval_set.json"
    log_path = eval_dir / "llm_addition.log"
    logger = setup_file_logger(log_path)

    judge = LlmSimilarityDecider(
        model=args.model,
        url=args.url,
        temperature=args.temperature,
        timeout=args.timeout,
        batch_size=args.batch_size,
    )

    base_seeds: List[str] = sorted(norm(s) for s in expansions.keys())
    augmented: Set[str] = set(base_seeds)
    added_by_seed: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    accepted_by_seed: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    verify_by_seed: Dict[str, Dict[str, List[str]]] = {}

    for seed, cand_list in expansions.items():
        if not isinstance(cand_list, list):
            continue
        accepted, reasons, dbg = select_with_llm_for_seed(
            judge,
            seed,
            cand_list,
            max_per_seed=args.max_per_seed
        )
        verify_by_seed[norm(seed)] = dbg.get("verification", {})
        for t in sorted(accepted):
            accepted_by_seed[norm(seed)].append((t, reasons.get(t, "llm:accepted")))
            if t not in augmented:
                augmented.add(t)
                added_by_seed[norm(seed)].append((t, reasons.get(t, "llm:accepted")))

    out_list = sorted(augmented | set(base_seeds)) if args.sort else list(augmented | set(base_seeds))
    save_json(json_out_path, {"seed_terms": out_list})

    # ---- Log summary ----
    logger.info("# Additions written by augment_seeds_with_llm")
    logger.info(f"# Expansions: {Path(args.expansions).resolve()}")
    logger.info(f"# Output dir: {eval_dir.resolve()}")
    logger.info(f"# JSON out  : {json_out_path.resolve()}")
    logger.info(f"# Model={args.model} URL={args.url} Temp={args.temperature} Timeout={args.timeout}s")
    logger.info(f"# BatchSize={args.batch_size} MaxPerSeed={args.max_per_seed} Sort={bool(args.sort)}")
    logger.info("## new to eval set")
    log_per_seed_lines(logger, added_by_seed)
    logger.info("")
    logger.info("## all accepted by LLM, including already in eval set")
    log_per_seed_lines(logger, accepted_by_seed)
    logger.info("")
    log_verification_issues(logger, verify_by_seed)
    logger.info("")
    total_added = sum(len(v) for v in added_by_seed.values())
    logger.info(f"# Original seeds (from expansion keys): {len(base_seeds)}")
    logger.info(f"# After augmentation (unique terms in eval set): {len(out_list)}")
    logger.info(f"# Total added from expansions via LLM decisions: {total_added}")

if __name__ == "__main__":
    main()
