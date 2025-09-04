#!/usr/bin/env python3
"""
Interactive seed+candidate CLI (STRICT schema, same Ollama flow) ✅
==================================================================

What this does
--------------
- Starts an interactive prompt.
- For each loop, you enter a SEED and a CANDIDATE.
- Sends the pair to the LLM using the SAME strict JSON schema, prompts, and
  Ollama /api/chat payload/format as your evaluator.
- Prints the model's returned STRICT JSON verbatim (optionally pretty).
- Shows a CLI spinner while the request is running.

Notes
-----
- Querying/sending to Ollama is UNCHANGED: same `format` JSON Schema, system/user
  prompts, and envelope parsing.
- `relation` is enabled (optional field in the schema).
- Optional anchors: pass a file via --anchors AND --use-anchors at startup to use
  the closure prompt that includes your anchor list.

Usage
-----
  python interactive_decide_cli.py \
      --model llama3.3:latest \
      --url http://localhost:11434/api/chat \
      --temperature 0.0 \
      --tokens 2048 \
      --pretty

With anchors:
  python interactive_decide_cli.py \
      --use-anchors \
      --anchors path/to/seeds_or_list.json

Inside the REPL:
  - Type the seed when prompted, then the candidate.
  - Press Enter on "seed>" to reuse the previous seed.
  - Commands: /q or /quit to exit, /help for help.

"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import requests

# -----------------------------
# Shared defaults (unchanged)
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
# LLM relation categories (unchanged)
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
# LLM client (Ollama) — STRICT single-candidate protocol (UNCHANGED flow)
# -----------------------------

class LlmSimilarityDecider:
    def __init__(
        self,
        model: str,
        url: str = "http://localhost:11434/api/chat",
        temperature: float = 0.0,
        tokens: int = 2048,
        timeout: int = 60,
        include_relation: bool = True,  # relation enabled for CLI
        global_context: str = GLOBAL_CONTEXT_DEFAULT,
        session: Optional[requests.Session] = None,
    ):
        self.url = url
        self.timeout = timeout
        self.global_context = (global_context or GLOBAL_CONTEXT_DEFAULT).strip()
        self.include_relation = bool(include_relation)
        self.session = session or requests.Session()

        # Same Ollama payload and options
        self.base_payload = {
            "model": model,
            "options": {"temperature": float(temperature), "num_ctx": tokens},
            "stream": False,
        }

        # Initial pass system prompt (unchanged)
        self.system_initial = (
            "You are a semantic similarity decider for short terms and underscore-separated MWEs.\n"
            f"DOMAIN CONTEXT:\n{self.global_context}\n\n"
            "TASK (STRICT):\n"
            "Given a SEED and EXACTLY ONE CANDIDATE, return STRICT JSON with keys {seed, decisions"
            + (", relation" if self.include_relation else "")
            + "}.\n"
            "The ONLY valid outputs for 'decisions' are either [] (reject) or [<exact candidate>] (accept).\n"
            "Do NOT alter, normalize, or invent strings. Return no explanations.\n\n"
            "ACCEPT if the candidate helps a user find/name/describe the same concept as the seed OR is a closely\n"
            "neighboring concept in this R-CPD domain, including ALL of the following relation types:\n"
            "  • hypernym (umbrella term of the seed)\n"
            "  • hyponym (more specific instance/type of the seed)\n"
            "  • holonym (a whole that includes the seed as a part/member)\n"
            "  • cohyponym (a sibling under the same umbrella as the seed)\n"
            "Also accept synonyms, near-synonyms, and morphological variants (noun/verb/adj forms of the same phenomenon).\n\n"
            + ("When you include 'relation', choose exactly one from: " + ", ".join(RELATION_ENUM) + ".\n" if self.include_relation else "")
        )

        # Closure pass system prompt (unchanged)
        self.system_closure = (
            "You are expanding a concept bucket defined by the FULL SET OF SEED TERMS (potentially ~2000 strings).\n"
            "You will be given:\n"
            " • SEED (the focal term),\n"
            " • VOCAB_BUCKET_SEEDS (a long list of seed terms acting as anchors/context), and\n"
            " • EXACTLY ONE REMAINING_CANDIDATE.\n"
            "Return STRICT JSON {seed, decisions"
            + (", relation" if self.include_relation else "")
            + "}.\n"
            "The ONLY valid outputs for 'decisions' are [] or [<exact candidate>].\n"
            "Apply the same acceptance rules (hypernym, hyponym, holonym, cohyponym, synonyms, morphology).\n\n"
            f"DOMAIN CONTEXT:\n{self.global_context}\n"
        )

    # Per-call JSON schema locked to the exact candidate (unchanged)
    def _build_single_candidate_schema(self, candidate: str) -> dict:
        props = {
            "seed": {"type": "string"},
            "decisions": {
                "oneOf": [
                    {"type": "array", "maxItems": 0},
                    {
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

    def _build_user_prompt_initial(self, seed: str, candidate: str) -> str:
        base = (
            f"SEED: {seed}\n"
            f"CANDIDATE (evaluate only this exact string; accept ⇒ return it, reject ⇒ return []):\n- {candidate}\n\n"
        )
        if self.include_relation:
            return base + 'Respond ONLY with JSON {"seed": <seed>, "decisions": [] or [<exact candidate>], optionally "relation": <one label>}.'
        else:
            return base + 'Respond ONLY with JSON {"seed": <seed>, "decisions": [] or [<exact candidate>]}.'

    def _build_user_prompt_closure(self, seed: str, anchors: List[str], candidate: str) -> str:
        anc_str = "\n".join(f"- {a}" for a in anchors) if anchors else "(none)"
        base = (
            f"SEED: {seed}\n"
            f"VOCAB_BUCKET_SEEDS (long list acting as anchors/context):\n{anc_str}\n\n"
            f"REMAINING_CANDIDATE (evaluate ONLY this exact string):\n- {candidate}\n\n"
        )
        if self.include_relation:
            return base + 'Respond ONLY with JSON {"seed": <seed>, "decisions": [] or [<exact candidate>], optionally "relation": <one label>}.'
        else:
            return base + 'Respond ONLY with JSON {"seed": <seed>, "decisions": [] or [<exact candidate>]}.'

    # (unchanged POST + response parsing envelope)
    def _post(self, payload: dict) -> dict:
        resp = self.session.post(self.url, headers={"Content-Type": "application/json"}, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        body = resp.json()
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

    # -------- RAW calls (return the model's STRICT JSON unchanged) --------

    def call_initial_raw(self, seed: str, candidate: str) -> dict:
        payload = dict(self.base_payload)
        payload["format"] = self._build_single_candidate_schema(candidate)
        payload["messages"] = [
            {"role": "system", "content": self.system_initial},
            {"role": "user", "content": self._build_user_prompt_initial(seed, candidate)},
        ]
        return self._post(payload)

    def call_with_anchors_raw(self, seed: str, anchors: List[str], candidate: str) -> dict:
        payload = dict(self.base_payload)
        payload["format"] = self._build_single_candidate_schema(candidate)
        payload["messages"] = [
            {"role": "system", "content": self.system_closure},
            {"role": "user", "content": self._build_user_prompt_closure(seed, anchors, candidate)},
        ]
        return self._post(payload)

# -----------------------------
# Spinner utility
# -----------------------------

class Spinner:
    """A lightweight CLI spinner shown while a task is running."""
    FRAMES = ["|", "/", "-", "\\"]

    def __init__(self, text: str = " contacting LLM… ", enabled: bool = True, interval: float = 0.12):
        self.text = text
        self.enabled = enabled
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not self.enabled or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        i = 0
        while not self._stop.is_set():
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r{frame}{self.text}")
            sys.stdout.flush()
            i += 1
            time.sleep(self.interval)
        # clear line
        sys.stdout.write("\r" + " " * (len(self.text) + 4) + "\r")
        sys.stdout.flush()

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None

# -----------------------------
# Helpers for anchors (optional)
# -----------------------------

def _load_anchors(path: Path) -> List[str]:
    """
    Accepts:
      - JSON object: use KEYS as anchors (like your full seed vocab)
      - JSON array : use items as anchors
      - TXT file   : one term per line
    """
    if not path.exists():
        raise FileNotFoundError(f"Anchors file not found: {path}")
    text = path.read_text(encoding="utf-8")
    # Try JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [str(k) for k in data.keys()]
        if isinstance(data, list):
            out = []
            for t in data:
                if isinstance(t, str):
                    tt = t.strip()
                    if tt:
                        out.append(tt)
            return out
    except Exception:
        pass
    # Fallback: TXT
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]

# -----------------------------
# Interactive loop
# -----------------------------

QUIT_TOKENS = {"/q", "/quit", "quit", "exit", ":q", ":qa"}

def interactive_loop(decider: LlmSimilarityDecider, anchors: Optional[List[str]], use_anchors: bool, pretty: bool, spinner_enabled: bool):
    print("Interactive mode. Type /help for help, /quit to exit.")
    last_seed: Optional[str] = None

    while True:
        try:
            seed = input("seed> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye!")
            return
        if not seed and last_seed:
            seed = last_seed
            print(f"(reusing seed: {seed})")
        elif not seed:
            # no previous seed, keep asking
            continue

        if seed.lower() in QUIT_TOKENS:
            print("bye!")
            return
        if seed.lower() in {"/help", "help", "h", "?"}:
            print("Commands:")
            print("  /q, /quit, quit, exit : exit")
            print("  (Enter on seed> to reuse the previous seed)")
            continue

        try:
            candidate = input("candidate> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye!")
            return
        if candidate.lower() in QUIT_TOKENS:
            print("bye!")
            return
        if not candidate:
            # empty candidate — ignore loop
            continue

        last_seed = seed

        spin = Spinner(enabled=spinner_enabled)
        try:
            spin.start()
            if use_anchors:
                if not anchors:
                    raise ValueError("Anchors requested but none loaded. Provide --anchors and --use-anchors.")
                out = decider.call_with_anchors_raw(seed, anchors, candidate)
            else:
                out = decider.call_initial_raw(seed, candidate)
        except Exception as e:
            spin.stop()
            err = {"error": str(e)}
            print(json.dumps(err, ensure_ascii=False))
            continue
        finally:
            spin.stop()

        # Print model output
        if pretty:
            print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(out, ensure_ascii=False))

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Interactive strict single-term decision via Ollama (prints model's STRICT JSON).")
    # LLM connection (unchanged behavior)
    ap.add_argument("--model", type=str, default="llama3.3:latest", help="Ollama model name")
    ap.add_argument("--url", type=str, default="http://localhost:11434/api/chat", help="Ollama chat endpoint")
    ap.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    ap.add_argument("--tokens", type=int, default=8196, help="LLM context tokens (num_ctx)")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    ap.add_argument("--global_context", type=str, default=GLOBAL_CONTEXT_DEFAULT,
        help="Domain context sentence(s) injected into prompts")

    # Optional anchors
    ap.add_argument("--use-anchors", action="store_true", help="Use closure prompt with anchors list")
    ap.add_argument("--anchors", type=str, default=None,
                    help="Path to anchors file: JSON object (keys used), JSON array, or TXT (one per line)")

    # Output
    ap.add_argument("--pretty", action="store_true", default=True, help="Pretty-print returned JSON")
    ap.add_argument("--no-spinner", action="store_true", help="Disable the CLI spinner")

    args = ap.parse_args()

    session = requests.Session()
    decider = LlmSimilarityDecider(
        model=args.model,
        url=args.url,
        temperature=args.temperature,
        tokens=args.tokens,
        timeout=args.timeout,
        include_relation=True,  # keep relation allowed in schema/prompts
        global_context=args.global_context,
        session=session,
    )

    anchors: Optional[List[str]] = None
    if args.anchors:
        anchors = _load_anchors(Path(args.anchors))

    try:
        interactive_loop(
            decider=decider,
            anchors=anchors,
            use_anchors=bool(args.use_anchors),
            pretty=bool(args.pretty),
            spinner_enabled=not args.no_spinner,
        )
    except KeyboardInterrupt:
        print("\nbye!")

if __name__ == "__main__":
    main()
