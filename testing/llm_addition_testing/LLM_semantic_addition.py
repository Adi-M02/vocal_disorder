#!/usr/bin/env python3
"""
Augment seed terms using an LLM (via Ollama) to judge semantic relatedness
between each seed and its expansion candidates, with LLM-based "closure" rounds
and a broad R-CPD-aware prompt.

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
      [--batch_size 80] [--max_per_seed 0] [--sort] \
      [--closure_iters 2] [--inclusion_bias lenient] \
      [--global_context "custom domain sentence here"]

Usage (debug mode; prints only one seed, writes nothing):
  python augment_seeds_with_llm.py \
      --expansions path/to/expansions.json \
      --model llama3.3:latest \
      --debug_seed community
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
    Provides two modes:
      - initial pass (seed + candidates)
      - closure pass (seed + anchors + remaining candidates)
    """

    def __init__(
        self,
        model: str,
        url: str = "http://localhost:11434/api/chat",
        temperature: float = 0.0,
        timeout: int = 60,
        batch_size: int = 80,
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
    ):
        self.url = url
        self.timeout = timeout
        self.batch_size = max(1, int(batch_size))
        self.inclusion_bias = inclusion_bias.lower().strip()
        self.global_context = global_context.strip()
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

        # Bias line
        bias_line = (
            "Err slightly on the INCLUSIVE side when a candidate would help a user find/name/describe the same concept "
            "or closely neighboring concepts within this domain (medical terms, diagnostics, related symptoms, emotions, "
            "actions/behaviors, logistics, and community/platform terms)."
            if self.inclusion_bias == "lenient" else
            "Be reasonably inclusive across the listed categories, but avoid broad unrelated associations."
        )

        # === System prompt: initial pass (broad, domain-aware) ===
        self.system_initial = (
            "You are a semantic similarity decision maker for short terms and underscore-separated MWEs.\n"
            f"DOMAIN CONTEXT:\n{self.global_context}\n\n"
            "TASK:\n"
            "Given a SEED and a list of CANDIDATES, return the subset of candidates that are semantically similar to "
            "the seed in a practical vocabulary-expansion sense. Accept items that a user would reasonably expect to see "
            "in the same bucket as the seed on R-CPD discussions, including:\n"
            " • Medical terminology and anatomy/physiology terms related to the seed\n"
            " • Diagnostics and tests (e.g., manometry, FEES, barium swallow, endoscopy), ENT-related visits\n"
            " • Interventions/treatments (e.g., botox injection to the cricopharyngeus/UES, dilation, therapy)\n"
            " • Related symptoms and bodily sensations (e.g., chest pressure, bloating, gurgling, hiccups)\n"
            " • Patient emotions/experiences (e.g., anxiety, embarrassment, frustration, relief, validation)\n"
            " • Actions/behaviors (e.g., massage, breathing techniques, carbonation tests, diet changes, booking appointments)\n"
            " • Logistics (e.g., referrals, insurance, clinic names) and community/platform references\n"
            "Treat navigation phrases (e.g., this_subreddit, this_thread, find_this_community) and platform names "
            "(reddit, tiktok, instagram) as valid when they point to the same community/topic as the seed.\n\n"
            "CRITICAL:\n"
            " • Consider candidates EXACTLY as given; do NOT normalize, spell-correct, or modify text (even r/noburp).\n"
            " • Return STRICT JSON ONLY with keys: seed, decisions.\n"
            " • decisions MUST be a list of the accepted candidate STRINGS, UNCHANGED from input.\n"
            f" • {bias_line}\n"
        )

        # === System prompt: closure pass (seed + anchors; same broad categories) ===
        self.system_closure = (
            "You are expanding a concept bucket. You are given a SEED and ANCHORS (already accepted examples). "
            "From REMAINING_CANDIDATES, select additional strings that clearly belong to the SAME bucket as the anchors "
            "with respect to the seed within the R-CPD domain.\n\n"
            f"DOMAIN CONTEXT:\n{self.global_context}\n\n"
            "VALID ADDITIONS OFTEN INCLUDE:\n"
            " • Medical/anatomy terms overlapping the seed’s concept\n"
            " • Diagnostics/tests, encounters, specialist terms\n"
            " • Interventions/treatments, therapy modalities\n"
            " • Related symptoms/sensations and common lay phrasings\n"
            " • Emotions/psychosocial terms typical in patient talk\n"
            " • Actions/behaviors users take, daily-life impacts, self-care\n"
            " • Logistics (insurance, referrals, clinic) and community/platform/navigation terms\n\n"
            "CRITICAL:\n"
            " • Choose ONLY from REMAINING_CANDIDATES and return them UNCHANGED.\n"
            " • Return STRICT JSON ONLY with keys: seed, decisions (list of strings).\n"
        )

    def _build_user_prompt_initial(self, seed: str, candidates: List[str]) -> str:
        cand_str = "\n".join(f"- {c}" for c in candidates)
        return (
            f"SEED: {seed}\n"
            f"CANDIDATES (one per line; return accepted ones unchanged):\n{cand_str}\n\n"
            "Respond ONLY with JSON per the enforced schema."
        )

    def _build_user_prompt_closure(self, seed: str, anchors: List[str], candidates: List[str]) -> str:
        anc_str = "\n".join(f"- {a}" for a in anchors) if anchors else "(none)"
        cand_str = "\n".join(f"- {c}" for c in candidates)
        return (
            f"SEED: {seed}\n"
            f"ANCHORS (already accepted; examples of the bucket):\n{anc_str}\n\n"
            f"REMAINING_CANDIDATES (choose zero or more; return unchanged):\n{cand_str}\n\n"
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
        if not isinstance(raw, list):
            return [], "empty"
        if raw and isinstance(raw[0], dict):
            # Be forgiving if the model still returns objects; attempt to extract 'term'
            terms = []
            for item in raw:
                t = item.get("term") if isinstance(item, dict) else None
                if isinstance(t, str):
                    terms.append(t)
            return terms, "object->string"
        else:
            terms = [t for t in raw if isinstance(t, str)]
            return terms, "string"

    # ---- initial pass ----
    def judge_initial_batch(self, seed: str, candidates: List[str]) -> Tuple[List[str], Dict]:
        payload = dict(self.base_payload)
        payload["messages"] = [
            {"role": "system", "content": self.system_initial},
            {"role": "user", "content": self._build_user_prompt_initial(seed, candidates)},
        ]
        out = self._post(payload)
        terms, schema_used = self._coerce_terms(out)
        ver = verify_unchanged(candidates, terms)
        ver["schema_used"] = schema_used
        ver["phase"] = "initial"
        return terms, ver

    def judge_initial_all(self, seed: str, candidates: List[str]) -> Tuple[List[str], Dict]:
        if not candidates:
            return [], {"schema_used": "empty", "unknown_terms": [], "duplicates": [], "phase": "initial"}
        all_terms: List[str] = []
        agg_unknown: Set[str] = set()
        agg_dupes: Set[str] = set()
        schema_seen: Set[str] = set()

        N = len(candidates)
        for i in range(0, N, self.batch_size):
            chunk = candidates[i:i + self.batch_size]
            terms, ver = self.judge_initial_batch(seed, chunk)
            all_terms.extend(terms)
            agg_unknown |= set(ver.get("unknown_terms", []))
            agg_dupes   |= set(ver.get("duplicates", []))
            schema_seen.add(ver.get("schema_used", "unknown"))

        merged_ver = {
            "schema_used": "/".join(sorted(schema_seen)) if schema_seen else "empty",
            "unknown_terms": sorted(agg_unknown),
            "duplicates": sorted(agg_dupes),
            "phase": "initial",
        }
        return all_terms, merged_ver

    # ---- closure pass ----
    def judge_closure_batch(self, seed: str, anchors: List[str], candidates: List[str]) -> Tuple[List[str], Dict]:
        payload = dict(self.base_payload)
        payload["messages"] = [
            {"role": "system", "content": self.system_closure},
            {"role": "user", "content": self._build_user_prompt_closure(seed, anchors, candidates)},
        ]
        out = self._post(payload)
        terms, schema_used = self._coerce_terms(out)
        ver = verify_unchanged(candidates, terms)
        ver["schema_used"] = schema_used
        ver["phase"] = "closure"
        return terms, ver

    def judge_closure_all(self, seed: str, anchors: List[str], candidates: List[str]) -> Tuple[List[str], Dict]:
        if not candidates:
            return [], {"schema_used": "empty", "unknown_terms": [], "duplicates": [], "phase": "closure"}
        all_terms: List[str] = []
        agg_unknown: Set[str] = set()
        agg_dupes: Set[str] = set()
        schema_seen: Set[str] = set()

        N = len(candidates)
        for i in range(0, N, self.batch_size):
            chunk = candidates[i:i + self.batch_size]
            terms, ver = self.judge_closure_batch(seed, anchors, chunk)
            all_terms.extend(terms)
            agg_unknown |= set(ver.get("unknown_terms", []))
            agg_dupes   |= set(ver.get("duplicates", []))
            schema_seen.add(ver.get("schema_used", "unknown"))

        merged_ver = {
            "schema_used": "/".join(sorted(schema_seen)) if schema_seen else "empty",
            "unknown_terms": sorted(agg_unknown),
            "duplicates": sorted(agg_dupes),
            "phase": "closure",
        }
        return all_terms, merged_ver

# -----------------------------
# Selection logic per seed (LLM-based with closure)
# -----------------------------
def select_with_llm_for_seed(
    judge: LlmSimilarityDecider,
    seed: str,
    cand_list: List[str],
    *,
    max_per_seed: int = 0,
    closure_iters: int = 2
) -> Tuple[Set[str], Dict[str, str], Dict]:
    """
    Returns (accepted, reasons_map, dbg).
    Steps:
      1) Initial pass (seed vs candidates).
      2) closure_iters rounds: expand from accepted anchors into remaining candidates.
    """
    candidates = [t for t in cand_list if isinstance(t, str) and t.strip()]
    cand_set_all = set(candidates)

    # 1) Initial
    init_terms, ver0 = judge.judge_initial_all(seed, candidates)
    accepted = set(t for t in init_terms if t in cand_set_all)

    # 2) Closure rounds
    closure_details = []
    verifications = [ver0]
    anchors = sorted(accepted)
    remaining = sorted(cand_set_all - accepted)

    for round_idx in range(1, max(0, int(closure_iters)) + 1):
        if not remaining:
            break
        add_terms, veri = judge.judge_closure_all(seed, anchors, remaining)
        verifications.append(veri)
        added = set(t for t in add_terms if t in remaining)
        closure_details.append({"round": round_idx, "added": sorted(added)})
        if not added:
            break
        accepted |= added
        anchors = sorted(accepted)  # grow anchors
        remaining = sorted(cand_set_all - accepted)

    # Cap per seed if requested
    if max_per_seed > 0 and len(accepted) > max_per_seed:
        accepted = set(sorted(list(accepted))[:max_per_seed])

    reasons = {t: "llm:accepted" for t in accepted}
    dbg = {
        "initial_accepted": sorted(set(init_terms) & cand_set_all),
        "closure": closure_details,
        "accepted": sorted(accepted),
        "verification": verifications,  # list of per-phase verifications
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

def log_verification_issues(logger: logging.Logger, verify_map: Dict[str, List[Dict[str, List[str]]]]):
    logger.info("## verification issues (terms must be returned unchanged)")
    any_issues = False
    for seed in sorted(verify_map.keys()):
        per_seed = verify_map[seed] or []
        for vi in per_seed:
            if vi.get("unknown_terms") or vi.get("duplicates"):
                any_issues = True
                break
    if not any_issues:
        logger.info("(none)")
        return
    for seed in sorted(verify_map.keys()):
        per_seed = verify_map[seed] or []
        phases_s = []
        for vi in per_seed:
            sch = vi.get("schema_used", "unknown")
            phase = vi.get("phase", "?")
            unk = vi.get("unknown_terms", [])
            dup = vi.get("duplicates", [])
            parts = [f"{phase}:schema={sch}"]
            if unk: parts.append(f"unknown={','.join(unk[:20])}{'…' if len(unk)>20 else ''}")
            if dup: parts.append(f"dupes={','.join(dup)}")
            phases_s.append(" | ".join(parts))
        logger.info(f"{seed} :: {' || '.join(phases_s)}")

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
        inclusion_bias=args.inclusion_bias,
        global_context=args.global_context,
    )
    accepted, reasons, dbg = select_with_llm_for_seed(
        judge,
        seed,
        cand_list,
        max_per_seed=args.max_per_seed,
        closure_iters=args.closure_iters
    )

    print("=== DEBUG MODE (LLM with closure) ===")
    print(f"Seed               : {seed}")
    print(f"Model / URL        : {args.model} / {args.url}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Inclusion bias     : {args.inclusion_bias}")
    print(f"Closure iters      : {args.closure_iters}")
    print("")
    print(f"Initial accepted [{len(dbg['initial_accepted'])}]: {join_uniq_sorted(dbg['initial_accepted']) or '(none)'}")
    for rd in dbg["closure"]:
        print(f"Closure round {rd['round']:>2d} added [{len(rd['added'])}]: {', '.join(rd['added']) or '(none)'}")
    print("")
    print(f"FINAL accepted [{len(accepted)}]: {join_uniq_sorted(accepted) or '(none)'}")
    print("")
    print("Verification:")
    for vi in dbg["verification"]:
        phase = vi.get("phase","?")
        print(f"  [{phase}] schema_used={vi.get('schema_used','unknown')}, "
              f"unknown_terms={', '.join(vi.get('unknown_terms', [])) or '(none)'}, "
              f"duplicates={', '.join(vi.get('duplicates', [])) or '(none)'}")
    print("")
    print("Reasons (first 25):")
    for i, (t, r) in enumerate(sorted(reasons.items())):
        if i >= 25: break
        print(f"  {t:<30s} -> {r}")
    print("==============================")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Augment seed terms using an Ollama LLM to judge semantic relatedness, with closure (R-CPD aware).")
    ap.add_argument("--expansions", required=True, help="Path to expansions JSON {seed: [terms]}")
    ap.add_argument("--outdir", help="Directory for outputs (ignored in --debug_seed mode)")
    ap.add_argument("--model", type=str, default="llama3.3:latest", help="Ollama model name")
    ap.add_argument("--url", type=str, default="http://localhost:11434/api/chat", help="Ollama chat endpoint")
    ap.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    ap.add_argument("--batch_size", type=int, default=80, help="Candidates per LLM call")
    ap.add_argument("--max_per_seed", type=int, default=0, help="Cap accepted terms per seed (0=off)")
    ap.add_argument("--sort", action="store_true", help="Sort final seed list alphabetically")
    ap.add_argument("--closure_iters", type=int, default=2, help="Number of closure expansion rounds per seed")
    ap.add_argument("--inclusion_bias", type=str, choices=["normal","lenient"], default="lenient",
                    help="Bias prompts to be slightly more inclusive")
    ap.add_argument("--global_context", type=str, default=(
        "The domain is R-CPD (Retrograde Cricopharyngeus Dysfunction: inability to burp) on Reddit. "
        "Expansions may include medical terminology, anatomy/physiology terms (e.g., cricopharyngeus, UES, esophagus), "
        "diagnostics (ENT visits, manometry, FEES, barium swallow, endoscopy), interventions/treatments "
        "(botox to cricopharyngeus/UES, dilation, therapy), related symptoms (chest pressure, bloating, gurgling, hiccups, "
        "nausea, reflux-like sensations), patient emotions (anxiety, embarrassment, frustration, relief, validation), "
        "actions/behaviors (massage, breathing techniques, carbonation tests, dietary changes, booking appointments), "
        "logistics (insurance, referrals, clinic names), and community/platform references (subreddit, reddit, tiktok, instagram, threads, groups)."
    ), help="Short domain context sentence(s) injected into prompts")
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
        inclusion_bias=args.inclusion_bias,
        global_context=args.global_context,
    )

    base_seeds: List[str] = sorted(norm(s) for s in expansions.keys())
    augmented: Set[str] = set(base_seeds)
    added_by_seed: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    accepted_by_seed: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    verify_by_seed: Dict[str, List[Dict[str, List[str]]]] = {}

    for seed, cand_list in expansions.items():
        if not isinstance(cand_list, list):
            continue
        accepted, reasons, dbg = select_with_llm_for_seed(
            judge,
            seed,
            cand_list,
            max_per_seed=args.max_per_seed,
            closure_iters=args.closure_iters
        )
        verify_by_seed[norm(seed)] = dbg.get("verification", [])
        for t in sorted(accepted):
            accepted_by_seed[norm(seed)].append((t, reasons.get(t, "llm:accepted")))
            if t not in augmented:
                augmented.add(t)
                added_by_seed[norm(seed)].append((t, reasons.get(t, "llm:accepted")))

    out_list = sorted(augmented | set(base_seeds)) if args.sort else list(augmented | set(base_seeds))
    save_json(json_out_path, {"seed_terms": out_list})

    # ---- Log summary ----
    logger.info("# Additions written by augment_seeds_with_llm (with closure, R-CPD aware)")
    logger.info(f"# Expansions: {Path(args.expansions).resolve()}")
    logger.info(f"# Output dir: {eval_dir.resolve()}")
    logger.info(f"# JSON out  : {json_out_path.resolve()}")
    logger.info(f"# Model={args.model} URL={args.url} Temp={args.temperature} Timeout={args.timeout}s")
    logger.info(f"# BatchSize={args.batch_size} MaxPerSeed={args.max_per_seed} Sort={bool(args.sort)} "
                f"ClosureIters={args.closure_iters} InclusionBias={args.inclusion_bias}")
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
