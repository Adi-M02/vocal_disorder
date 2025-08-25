#!/usr/bin/env python3
"""
Single-term evaluator with WordNet-aware hints (STRICT one-candidate schema)
===========================================================================

What this does
--------------
- Evaluates **one expansion term at a time** against its seed using an LLM.
- Adds **WordNet signals** (hypernyms, hyponyms, holonyms, co-hyponyms, synonyms,
  derivationals, adjective neighbors) as *hints* to the LLM and (optionally) as an
  **override** if the model rejects but WordNet says it's strongly related.
- **Strict I/O schema per call** so the model can ONLY emit either [] (reject) or
  ["<exact candidate>"] (accept).
- Rich logging + verification:
  • Per-pair NDJSON audit: latency, attempts, schema used, seed echo, unknown/dupe issues,
    WordNet relations detected, and whether a WordNet override was used.
  • Per-seed NDJSON line: {seed, accepted:[...], checked_subset:bool, violations:[...]}.
  • Human-readable summary log with per-seed acceptance rates + error sections.

Outputs
-------
  • decisions.ndjson                (per-pair audit)
  • seeds_accepted.ndjson          (one line per seed with accepted list + subset check)
  • accepted_by_seed.json          { seed: [accepted terms] }
  • filtered_expansions.json       { seed: [accepted terms] } (same as above; mirrors input shape)
  • accepted_all_flat.json         [all accepted terms across all seeds, in encounter order]
  • accepted_aligned_by_seed.json  { seed: [accepted-or-empty-string aligned to input order] }
  • summary.json                   run stats + settings

CLI knobs (WordNet)
-------------------
  --hyper_depth, --hypo_depth, --hol_depth
  --cohypo_up, --cohypo_down
  --adj_hops
  --min_ancestor_depth
  --pos_gate_nouns (1/0)
  --accept_if_wordnet  (if set, accept when strong WordNet relation found, unless ultra-generic)

Example (debug a single pair)
-----------------------------
  python LLM_semantic_addition_single_wn.py \
      --expansions path/to/expansions.json \
      --debug_seed tachycardia --debug_term heart_condition \
      --accept_if_wordnet
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

# --- WordNet setup ---------------------------------------------------------
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.corpus import wordnet

# -----------------------------
# Helpers
# -----------------------------

def norm(tok: str) -> str:
    return tok.strip().lower()

def to_wordnet(tok: str) -> str:
    return norm(tok).replace(" ", "_")

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
# WordNet graph utilities
# -----------------------------

def _lemma_names(synsets: Iterable[wordnet.synset]) -> Set[str]:
    out = set()
    for ss in synsets:
        for l in ss.lemmas():
            out.add(l.name().replace("_", " ").lower())
    return out

def _derivational_from_synsets(synsets: Iterable[wordnet.synset]) -> Set[str]:
    out = set()
    for ss in synsets:
        for l in ss.lemmas():
            for d in l.derivationally_related_forms():
                out.add(d.name().replace("_", " ").lower())
    return out

def _bfs(start: Set[wordnet.synset], step_fn, depth: int) -> Set[wordnet.synset]:
    """BFS over synset graph up to 'depth' edges; returns all visited synsets (incl. start)."""
    if depth <= 0 or not start:
        return set(start)
    seen = set(start)
    frontier = set(start)
    for _ in range(depth):
        nxt = set()
        for ss in frontier:
            for nb in step_fn(ss):
                if nb not in seen:
                    seen.add(nb); nxt.add(nb)
        frontier = nxt
    return seen

def synsets_any(word: str) -> List[wordnet.synset]:
    return wordnet.synsets(to_wordnet(word))  # all POS

def synsets_pos(word: str, pos: str) -> List[wordnet.synset]:
    return wordnet.synsets(to_wordnet(word), pos=pos)

def synonyms(word: str) -> Set[str]:
    return _lemma_names(synsets_any(word))

def derivationals(word: str) -> Set[str]:
    return _derivational_from_synsets(synsets_any(word))

def hypernyms(word: str, depth: int = 1, *, min_ancestor_depth: int = 0) -> Set[str]:
    """N/V hypernyms up to depth hops. Filters ancestors by WordNet min_depth >= min_ancestor_depth."""
    ssets = [ss for ss in synsets_any(word) if ss.pos() in ("n", "v")]
    if not ssets or depth <= 0:
        return set()
    def step(ss):
        out = list(ss.hypernyms())
        if ss.pos() == "n":
            out += list(ss.instance_hypernyms())
        return out
    reached = _bfs(set(ssets), step, depth)
    if min_ancestor_depth > 0:
        reached = {ss for ss in reached if hasattr(ss, "min_depth") and ss.min_depth() >= min_ancestor_depth}
    return _lemma_names(reached)

def hyponyms(word: str, depth: int = 1) -> Set[str]:
    """N/V hyponyms up to depth hops."""
    ssets = [ss for ss in synsets_any(word) if ss.pos() in ("n", "v")]
    if not ssets or depth <= 0:
        return set()
    def step(ss):
        return list(ss.hyponyms())
    reached = _bfs(set(ssets), step, depth)
    return _lemma_names(reached)

def holonyms(word: str, depth: int = 1) -> Set[str]:
    """Holonyms (member/part/substance) for NOUN senses up to depth hops."""
    ssets = synsets_pos(word, wordnet.NOUN)
    if not ssets or depth <= 0:
        return set()
    def step(ss):
        return list(ss.member_holonyms()) + list(ss.part_holonyms()) + list(ss.substance_holonyms())
    reached = _bfs(set(ssets), step, depth)
    return _lemma_names(reached)

def cohyponyms(word: str, up: int = 1, down: int = 1, *, noun_only: bool = True,
               min_ancestor_depth: int = 0) -> Set[str]:
    """
    Siblings via hypernym(s): go UP 'up' hypernym hops from the seed's synsets,
    then DOWN 'down' hyponym hops; ancestors are filtered by min_depth if provided.
    Excludes the seed's own lemmas (keeps true siblings).
    """
    ssets = [ss for ss in synsets_any(word) if (ss.pos()=="n" if noun_only else ss.pos() in ("n","v"))]
    if not ssets or up <= 0 or down <= 0:
        return set()
    def up_step(ss):
        out = list(ss.hypernyms())
        if ss.pos() == "n":
            out += list(ss.instance_hypernyms())
        return out
    ancestors = _bfs(set(ssets), up_step, up)
    if min_ancestor_depth > 0:
        ancestors = {a for a in ancestors if hasattr(a, "min_depth") and a.min_depth() >= min_ancestor_depth}
    def down_step(ss):
        return list(ss.hyponyms())
    desc = set()
    for a in ancestors:
        desc |= _bfs({a}, down_step, down)
    names = _lemma_names(desc)
    seed_lemmas = _lemma_names(ssets)
    return {w for w in names if w not in seed_lemmas}

def adjective_neighbors(seeds: Iterable[str], hops: int = 2) -> Set[str]:
    """
    Build an adjective neighborhood by walking 'similar_tos' and 'also_sees'
    from ADJ synsets of (seed, seed+'ed', seed+'ing', and each seed's synonyms).
    """
    if hops <= 0:
        return set()
    start_synsets = set()
    for s in seeds:
        forms = {s, f"{s}ed", f"{s}ing"} | synonyms(s)
        for f in forms:
            for ss in synsets_pos(f, wordnet.ADJ):
                start_synsets.add(ss)
    def step(ss):
        return list(ss.similar_tos()) + list(ss.also_sees())
    reached = _bfs(start_synsets, step, depth=hops)
    return _lemma_names(reached)

def dominant_pos_is_noun(word: str) -> bool:
    """Return True if the seed's dominant POS (by synset count) is noun."""
    ss = synsets_any(word)
    if not ss:
        return False
    from collections import Counter
    cnt = Counter(s.pos() for s in ss)
    noun_count = cnt.get('n', 0)
    other = cnt.get('v', 0) + cnt.get('a', 0) + cnt.get('s', 0) + cnt.get('r', 0)
    return noun_count >= other

def has_noun_sense(word: str) -> bool:
    """Does the candidate have at least one noun synset?"""
    return any(ss.pos() == 'n' for ss in synsets_any(word))

# Build a WordNet neighborhood for a *set* of anchors (usually just {seed})
def expand_wordnet_neighborhood(
    anchors: Iterable[str],
    *,
    include_syn=True,
    include_deriv=True,
    hyper_depth=1,
    hypo_depth=1,
    hol_depth=1,
    cohypo_up=0,
    cohypo_down=0,
    adj_hops=2,
    min_ancestor_depth=0,
) -> Set[str]:
    anchors = {norm(a) for a in anchors}
    out: Set[str] = set()

    if include_syn:
        for a in anchors:
            out |= synonyms(a)
    if include_deriv:
        for a in anchors:
            out |= derivationals(a)
    if hyper_depth > 0:
        for a in anchors:
            out |= hypernyms(a, hyper_depth, min_ancestor_depth=min_ancestor_depth)
    if hypo_depth > 0:
        for a in anchors:
            out |= hyponyms(a, hypo_depth)
    if hol_depth > 0:
        for a in anchors:
            out |= holonyms(a, hol_depth)
    if cohypo_up > 0 and cohypo_down > 0:
        for a in anchors:
            out |= cohyponyms(a, up=cohypo_up, down=cohypo_down, noun_only=True, min_ancestor_depth=min_ancestor_depth)
    if adj_hops > 0:
        out |= adjective_neighbors(anchors, hops=adj_hops)

    return {norm(x) for x in out}

# For a seed→candidate pair, detect *which* relations fire (labels)
def wordnet_relations_for_pair(
    seed: str,
    candidate: str,
    *,
    hyper_depth: int,
    hypo_depth: int,
    hol_depth: int,
    cohypo_up: int,
    cohypo_down: int,
    adj_hops: int,
    min_ancestor_depth: int,
    pos_gate_nouns: bool,
) -> List[str]:
    s = norm(seed)
    c = norm(candidate)

    relations: List[str] = []

    # POS gate: if seed is dominantly noun, only consider noun-sensitive relations for candidates
    if pos_gate_nouns and dominant_pos_is_noun(s):
        if not has_noun_sense(c):
            return relations  # nothing (we'll still allow LLM to decide)

    # Compute each relation set and test membership
    if c in synonyms(s):
        relations.append("synonym")
    if c in derivationals(s):
        relations.append("derivational")
    if hyper_depth > 0 and c in hypernyms(s, hyper_depth, min_ancestor_depth=min_ancestor_depth):
        relations.append("hypernym")
    if hypo_depth > 0 and c in hyponyms(s, hypo_depth):
        relations.append("hyponym")
    if hol_depth > 0 and c in holonyms(s, hol_depth):
        relations.append("holonym")
    if cohypo_up > 0 and cohypo_down > 0 and c in cohyponyms(s, up=cohypo_up, down=cohypo_down, noun_only=True, min_ancestor_depth=min_ancestor_depth):
        relations.append("cohyponym")
    if adj_hops > 0 and c in adjective_neighbors({s}, hops=adj_hops):
        relations.append("adjacent_adj")

    return relations

ULTRA_GENERIC = {
    "medical_condition", "health_problem", "health_issue", "problem", "issue",
    "condition", "symptom", "thing", "stuff", "pain", "disorder"
}

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
        global_context: str = (
            "The domain is R-CPD (Retrograde Cricopharyngeus Dysfunction: inability to burp) on Reddit. "
            "Expansions may include medical terminology, anatomy/physiology terms (e.g., cricopharyngeus, UES, esophagus), "
            "diagnostics (ENT visits, manometry, FEES, barium swallow, endoscopy), interventions/treatments "
            "(botox to cricopharyngeus/UES, dilation, therapy), related symptoms (chest pressure, bloating, gurgling, hiccups, "
            "nausea, reflux-like sensations), patient emotions (anxiety, embarrassment, frustration, relief, validation), "
            "actions/behaviors (massage, breathing techniques, carbonation tests, dietary changes, booking appointments), "
            "logistics (insurance, referrals, clinic names), and community/platform references (subreddit, reddit, tiktok, instagram), "
            "abbreviations of common medical terms, and umbrella/hypernym lay terms (e.g., heart_condition, throat_condition, stomach_issue)."
        ),
        session: Optional[requests.Session] = None,
    ):
        self.url = url
        self.timeout = timeout
        self.global_context = global_context.strip()
        self.headers = {"Content-Type": "application/json"}
        self.session = session or requests.Session()

        # Per-call schema is injected per candidate (no static format here)
        self.base_payload = {
            "model": model,
            "options": {"temperature": float(temperature)},
            "stream": False,
        }

        # Initial pass: single candidate, strict schema + WordNet acceptance guidance
        self.system_initial = (
            "You are a semantic similarity decider for short terms and underscore-separated MWEs.\n"
            f"DOMAIN CONTEXT:\n{self.global_context}\n\n"
            "TASK (STRICT):\n"
            "Given a SEED and EXACTLY ONE CANDIDATE, return STRICT JSON with keys {seed, decisions}.\n"
            "The ONLY valid outputs are either [] (reject) or [<exact candidate>] (accept).\n"
            "Do NOT alter, normalize, or invent strings. Return no explanations.\n\n"
            "CRITICAL:\n"
            " • Consider text EXACTLY as given; do NOT modify.\n"
            " • Return STRICT JSON ONLY with keys: seed, decisions.\n"
            " • Accept the candidate if it helps a user find/name/describe the same concept as the seed,\n"
            "   OR a closely neighboring concept in this domain (medical terms, diagnostics, related symptoms, emotions,\n"
            "   actions/behaviors, logistics, community/platform terms, abbreviations).\n"
            " • Accept **morphological family** variants describing the same phenomenon across noun/verb/adjective forms.\n"
            " • When WORDNET SIGNALS indicate the candidate is a **hypernym/umbrella**, **hyponym/specific**, **holonym**, or **co-hyponym**\n"
            "   of the seed, accept (unless the candidate is ultra-generic like 'medical_condition' or 'health_problem').\n\n"
            "EXAMPLES (read-only):\n"
            "  - SEED: tachycardia | CANDIDATE: heart_condition   → [\"heart_condition\"]  (hypernym/umbrella)\n"
            "  - SEED: reflux      | CANDIDATE: digestive_disorder→ [\"digestive_disorder\"] (hypernym)\n"
            "  - SEED: reflux      | CANDIDATE: medical_condition → []                     (ultra-generic)\n"
            "  - SEED: nauseous    | CANDIDATE: nausea            → [\"nausea\"]            (morphology/near-synonym)\n"
        )

        # Closure pass: bucket = FULL seed vocabulary (potentially ~2000 strings) + WordNet guidance
        self.system_closure = (
            "You are expanding a concept bucket defined by the FULL SET OF SEED TERMS (potentially ~2000 strings).\n"
            "You will be given:\n"
            " • SEED (the focal term),\n"
            " • VOCAB_BUCKET_SEEDS (a long list of seed terms acting as anchors/context and relevant terms to match to), and\n"
            " • EXACTLY ONE REMAINING_CANDIDATE.\n"
            "Return STRICT JSON {seed, decisions}. The ONLY valid outputs are [] or [<exact candidate>].\n"
            "Decide based on semantic relatedness to the SEED, informed by the overall bucket of seeds within the R-CPD domain.\n"
            "Accept morphological family variants, and apply the same **WordNet signals** rule (hypernym/hyponym/holonym/co-hyponym) unless ultra-generic.\n\n"
            f"DOMAIN CONTEXT:\n{self.global_context}\n"
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

    def _build_user_prompt_initial(self, seed: str, candidate: str, wn_signals: List[str]) -> str:
        wn_str = ""
        if wn_signals:
            bullets = "\n".join(f"- {sig}" for sig in wn_signals)
            wn_str = f"\nWORDNET SIGNALS (seed→candidate):\n{bullets}\n"
        return (
            f"SEED: {seed}\n"
            f"CANDIDATE (evaluate only this exact string; accept ⇒ return it, reject ⇒ return []):\n- {candidate}\n"
            f"{wn_str}"
            "Respond ONLY with JSON {\"seed\": <seed>, \"decisions\": [] or [<exact candidate>]}."
        )

    def _build_user_prompt_closure(self, seed: str, anchors: List[str], candidate: str, wn_signals: List[str]) -> str:
        anc_str = "\n".join(f"- {a}" for a in anchors) if anchors else "(none)"
        wn_str = ""
        if wn_signals:
            bullets = "\n".join(f"- {sig}" for sig in wn_signals)
            wn_str = f"\nWORDNET SIGNALS (seed→candidate):\n{bullets}\n"
        return (
            f"SEED: {seed}\n"
            f"VOCAB_BUCKET_SEEDS (long list acting as anchors/context):\n{anc_str}\n\n"
            f"REMAINING_CANDIDATE (evaluate ONLY this exact string):\n- {candidate}\n"
            f"{wn_str}"
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
            # Defensive: convert list of objects with {term: ...} into strings
            terms: List[str] = []
            for item in raw:
                t = item.get("term") if isinstance(item, dict) else None
                if isinstance(t, str):
                    terms.append(t)
            return terms, "object->string"
        else:
            return [t for t in raw if isinstance(t, str)], "string"

    # Single-candidate calls ----------------------------------------------
    def judge_single_initial(self, seed: str, candidate: str, wn_signals: List[str]) -> Tuple[bool, Dict]:
        payload = dict(self.base_payload)
        payload["format"] = self._build_single_candidate_schema(candidate)
        payload["messages"] = [
            {"role": "system", "content": self.system_initial},
            {"role": "user", "content": self._build_user_prompt_initial(seed, candidate, wn_signals)},
        ]
        out = self._post(payload)
        terms, schema_used = self._coerce_terms(out)
        returned_seed = out.get("seed") if isinstance(out, dict) else None  # seed echo check
        ver = verify_unchanged([candidate], terms)
        ver["schema_used"] = schema_used
        ver["phase"] = "initial"
        ver["seed_echo_ok"] = (returned_seed == seed)
        accepted = (len(terms) == 1 and terms[0] == candidate)
        schema_ok = (len(terms) in (0, 1)) and not ver["unknown_terms"] and not ver["duplicates"] and ver["seed_echo_ok"]
        ver["schema_ok"] = schema_ok
        return accepted, ver

    def judge_single_with_anchors(self, seed: str, anchors: List[str], candidate: str, wn_signals: List[str]) -> Tuple[bool, Dict]:
        payload = dict(self.base_payload)
        payload["format"] = self._build_single_candidate_schema(candidate)
        payload["messages"] = [
            {"role": "system", "content": self.system_closure},
            {"role": "user", "content": self._build_user_prompt_closure(seed, anchors, candidate, wn_signals)},
        ]
        out = self._post(payload)
        terms, schema_used = self._coerce_terms(out)
        returned_seed = out.get("seed") if isinstance(out, dict) else None
        ver = verify_unchanged([candidate], terms)
        ver["schema_used"] = schema_used
        ver["phase"] = "closure"
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
    prompt_type: str  # "initial" | "closure" | "shortcut"
    schema_used: str = "unknown"
    unknown_terms: List[str] = field(default_factory=list)
    duplicates: List[str] = field(default_factory=list)
    attempts: int = 1
    latency_ms: int = 0
    error: Optional[str] = None
    # WordNet logging
    wn_signals: List[str] = field(default_factory=list)
    wn_override_used: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

# -----------------------------
# Runner (sequential across seeds; no multi-seed concurrency)
# -----------------------------

class SingleTermRunner:
    def __init__(self, args: argparse.Namespace, global_anchors: Optional[List[str]] = None):
        self.args = args
        self.session = requests.Session()
        self.judge = LlmSimilarityDecider(
            model=args.model,
            url=args.url,
            temperature=args.temperature,
            timeout=args.timeout,
            global_context=args.global_context,
            session=self.session,
        )
        # Global anchors = entire seed vocabulary when --use_anchors is passed
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

    def _wordnet_signals_for_pair(self, seed: str, candidate: str) -> List[str]:
        return wordnet_relations_for_pair(
            seed, candidate,
            hyper_depth=self.args.hyper_depth,
            hypo_depth=self.args.hypo_depth,
            hol_depth=self.args.hol_depth,
            cohypo_up=self.args.cohypo_up,
            cohypo_down=self.args.cohypo_down,
            adj_hops=self.args.adj_hops,
            min_ancestor_depth=self.args.min_ancestor_depth,
            pos_gate_nouns=bool(self.args.pos_gate_nouns),
        )

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
                wn_signals=["identity"],
                wn_override_used=False,
            )
            if self.args.cache_path:
                self._append_cache(self.args.cache_path, rec)
            return rec

        # If --use_anchors, always use the (large) global seed vocabulary as anchors
        use_anchors = bool(self.args.use_anchors and anchors)

        # Compute WordNet signals for this pair
        wn_signals = self._wordnet_signals_for_pair(seed, candidate)

        # Call LLM (with WordNet hints included in the user prompt)
        if use_anchors:
            ok, ver, ms, err = self._call_with_retries(self.judge.judge_single_with_anchors, seed, anchors or [], candidate, wn_signals)
            prompt_type = "closure"
        else:
            ok, ver, ms, err = self._call_with_retries(self.judge.judge_single_initial, seed, candidate, wn_signals)
            prompt_type = "initial"

        # Decide final acceptance with optional WordNet override
        wn_override_used = False
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
                wn_signals=wn_signals,
                wn_override_used=False,
            )
        else:
            schema_ok = ver.get("schema_ok", True)
            decision = None
            accepted = ok

            # Optional WordNet override: only if LLM rejected AND we have meaningful signals AND not ultra-generic
            if (not accepted) and self.args.accept_if_wordnet and wn_signals and (norm(candidate) not in ULTRA_GENERIC):
                accepted = True
                wn_override_used = True

            if accepted:
                decision = "accept"
            elif (not schema_ok) or ver.get("unknown_terms") or ver.get("duplicates"):
                decision = "unknown_mismatch"
            else:
                decision = "reject"

            rec = DecisionRecord(
                seed=seed,
                candidate=candidate,
                accepted=accepted,
                decision=decision,
                prompt_type=prompt_type,
                schema_used=ver.get("schema_used", "unknown"),
                unknown_terms=ver.get("unknown_terms", []),
                duplicates=ver.get("duplicates", []),
                attempts=1,
                latency_ms=ms,
                error=None,
                wn_signals=wn_signals,
                wn_override_used=wn_override_used,
            )

        if self.args.cache_path:
            self._append_cache(self.args.cache_path, rec)
        return rec

    # Per-seed sequential flow; **no** within-seed or cross-seed concurrency
    def process_seed(self, seed: str, candidates: List[str]) -> Tuple[Set[str], List[DecisionRecord], List[str]]:
        accepted: Set[str] = set()
        records: List[DecisionRecord] = []
        aligned: List[str] = []  # same length/order as input candidates; accepted term or ""

        # Decide which anchors to use: global seed vocab (if --use_anchors) vs none
        anchors_ctx: Optional[List[str]] = self.global_anchors if self.args.use_anchors and self.global_anchors else None

        for cand in candidates:
            if not isinstance(cand, str) or not cand:
                # Emit an error record for bad candidate type
                records.append(DecisionRecord(seed=seed, candidate=str(cand), accepted=False, decision="error", prompt_type="initial", error="non-string or empty candidate"))
                aligned.append("")
                continue

            rec = self.eval_pair(seed, cand, anchors=anchors_ctx)
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

    wn_any = sum(1 for r in all_records if r.wn_signals)
    wn_over = sum(1 for r in all_records if r.wn_override_used)

    logger.info("# Single-term LLM evaluation summary (with WordNet signals)")
    logger.info(f"Total evals: {total} | accept={acc} reject={rej} unknown_mismatch={unk} error={err}")
    logger.info(f"WordNet signals present in {wn_any} pairs; WordNet overrides used in {wn_over} pairs.")
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
        wn_any_s = sum(1 for r in seed_recs if r.wn_signals)
        wn_over_s = sum(1 for r in seed_recs if r.wn_override_used)
        logger.info(
            f"{seed:<28s} | total={len(seed_recs):4d} accept={acc_s:4d} reject={rej_s:4d} unknown={unk_s:3d} error={err_s:3d} "
            f"WN_signals={wn_any_s:4d} WN_overrides={wn_over_s:4d}"
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
# Main (sequential across seeds)
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Single-term LLM judgments for seed expansions (STRICT schema, R-CPD aware, WordNet-guided).")
    ap.add_argument("--expansions", required=True, help="Path to expansions JSON {seed: [terms]}")
    ap.add_argument("--outdir", required=False, help="Directory for outputs (ignored in debug mode)")

    # LLM
    ap.add_argument("--model", type=str, default="llama3.3:latest", help="Ollama model name")
    ap.add_argument("--url", type=str, default="http://localhost:11434/api/chat", help="Ollama chat endpoint")
    ap.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    ap.add_argument("--global_context", type=str, default=(
        "The domain is R-CPD (Retrograde Cricopharyngeus Dysfunction: inability to burp) on Reddit. "
        "Expansions may include medical terminology, anatomy/physiology terms (e.g., cricopharyngeus, UES, esophagus), "
        "diagnostics (ENT visits, manometry, FEES, barium swallow, endoscopy), interventions/treatments (botox to cricopharyngeus/UES, dilation, therapy), "
        "related symptoms (chest pressure, bloating, gurgling, hiccups, nausea, reflux-like sensations), patient emotions (anxiety, embarrassment, frustration, relief, validation), "
        "actions/behaviors (massage, breathing techniques, carbonation tests, dietary changes, booking appointments), logistics (insurance, referrals, clinic names), "
        "community/platform references, abbreviations, and lay umbrella terms."),
        help="Short domain context sentence(s) injected into prompts",
    )

    # Execution controls
    ap.add_argument("--retries", type=int, default=2, help="Retry count on request/parse errors (exponential backoff)")
    ap.add_argument("--sort", action="store_true", help="Sort final lists alphabetically where applicable")
    ap.add_argument("--auto_accept_if_equal", action="store_true", help="Auto-accept when candidate == seed")

    # Filtering / limits
    ap.add_argument("--seed_filter", type=str, nargs="*", default=None, help="Only process these seeds (exact match)")
    ap.add_argument("--limit_per_seed", type=int, default=0, help="Limit number of candidates per seed (0=all)")

    # Anchors
    ap.add_argument("--use_anchors", action="store_true",
                    help="If set, the closure prompt uses the FULL seed vocabulary (all seeds in the JSON) as anchors/context.")

    # WordNet controls
    ap.add_argument("--hyper_depth", type=int, default=2, help="Hypernym hops")
    ap.add_argument("--hypo_depth",  type=int, default=2, help="Hyponym hops")
    ap.add_argument("--hol_depth",   type=int, default=1, help="Holonym hops")
    ap.add_argument("--cohypo_up",   type=int, default=1, help="Co-hyponyms: hops up (0=off)")
    ap.add_argument("--cohypo_down", type=int, default=2, help="Co-hyponyms: hops down (from each ancestor)")
    ap.add_argument("--adj_hops",    type=int, default=1, help="Adjective neighbor hops")
    ap.add_argument("--min_ancestor_depth", type=int, default=0, help="Min WordNet min_depth for ancestor filters (0=off)")
    ap.add_argument("--pos_gate_nouns", type=int, default=1, help="If seed is dominantly noun, require candidate to have a noun sense (1/0)")
    ap.add_argument("--accept_if_wordnet", action="store_true", help="If set, accept on meaningful WordNet relation even when LLM rejects (guards ultra-generic)")

    # Caching
    ap.add_argument("--cache_path", type=str, default=None, help="Optional JSONL cache file for decisions (read+append)")

    # Debug
    ap.add_argument("--debug_seed", type=str, default=None, help="If set, evaluate only this seed and --debug_term")
    ap.add_argument("--debug_term", type=str, default=None, help="If set with --debug_seed, evaluate only this candidate")

    args = ap.parse_args()

    expansions = load_json(args.expansions)
    if not isinstance(expansions, dict):
        raise ValueError("Expansions JSON must be an object mapping seeds to lists of terms.")

    # Prepare global anchors (full seed vocabulary) if requested
    global_seed_vocab: List[str] = sorted([str(s) for s in expansions.keys()]) if args.use_anchors else []

    # --- DEBUG MODE: single pair ---
    if args.debug_seed and args.debug_term:
        runner = SingleTermRunner(args, global_anchors=global_seed_vocab)
        seed = args.debug_seed
        cand_list = expansions.get(seed)
        if not isinstance(cand_list, list):
            print(f"[warn] Seed '{seed}' not found or expansions not a list.")
            return
        if args.debug_term not in cand_list:
            print(f"[warn] Candidate '{args.debug_term}' is NOT in expansions for seed '{seed}'.")
        rec = runner.eval_pair(seed, args.debug_term, anchors=global_seed_vocab if args.use_anchors else None)
        print("=== DEBUG SINGLE-PAIR ===")
        print(json.dumps(asdict(rec), indent=2, ensure_ascii=False))
        return

    # --- NORMAL MODE (sequential across seeds) ---
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
    runner = SingleTermRunner(args, global_anchors=global_seed_vocab)

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
    logger.info("# Concurrency disabled: server serializes per-model requests; processing seeds sequentially.")
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
        "timeout": args.timeout,
        "use_anchors": bool(args.use_anchors),
        "retries": args.retries,
        "sorted": bool(args.sort),
        "accept_if_wordnet": bool(args.accept_if_wordnet),
        "wordnet_settings": {
            "hyper_depth": args.hyper_depth,
            "hypo_depth": args.hypo_depth,
            "hol_depth": args.hol_depth,
            "cohypo_up": args.cohypo_up,
            "cohypo_down": args.cohypo_down,
            "adj_hops": args.adj_hops,
            "min_ancestor_depth": args.min_ancestor_depth,
            "pos_gate_nouns": bool(args.pos_gate_nouns),
        },
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
    logger.info(f"# UseAnchors={bool(args.use_anchors)} Retries={args.retries} Sort={bool(args.sort)} AcceptIfWordNet={bool(args.accept_if_wordnet)}")
    logger.info("")
    log_summary(logger, all_records, accepted_by_seed)

if __name__ == "__main__":
    main()
