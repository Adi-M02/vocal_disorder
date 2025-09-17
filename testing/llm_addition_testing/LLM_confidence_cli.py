#!/usr/bin/env python3
"""
Interactive Y/N decider with probabilities via llama-server (/v1/chat/completions).

What it does
------------
- Prompts for SEED, then CANDIDATE (press Enter on SEED to reuse last).
- Sends a domain-specific system prompt (R-CPD context).
- Expects llama-server running with a JSON Schema that forces output "Y" or "N"
  (server started with: -jf yes_no_schema.json).
- Requests logprobs and extracts the step where Y/N is chosen.
- Prints the model's Y/N answer and normalized p(Y)/p(N).

Model handling
--------------
- If --model is omitted, the script auto-detects the first model from /v1/models
  on the same host/port (based on --url).

Usage
-----
  python yn_decider_cli.py \
    --url http://localhost:18080/v1/chat/completions \
    --pretty

Options
-------
- Keep temperature=0, top_k=0, top_p=1 for clean distributions.
- If Y/N sometimes missing from top_logprobs, raise --top-logprobs (e.g., 20).

"""

from __future__ import annotations
import argparse
import json
import math
import sys
from typing import Dict, Any, Optional, Tuple

import requests
from urllib.parse import urlparse, urlunparse

RCPD_DOMAIN_CONTEXT = (
    "The domain is R-CPD (Retrograde Cricopharyngeus Dysfunction: inability to burp) on Reddit. "
    "Expansions may include medical terminology, anatomy/physiology terms (e.g., cricopharyngeus, UES, esophagus), "
    "diagnostics (ENT visits, manometry, FEES, barium swallow, endoscopy), interventions/treatments "
    "(botox to cricopharyngeus/UES, dilation, therapy), related symptoms (chest pressure, bloating, gurgling, hiccups, "
    "nausea, reflux-like sensations), patient emotions (anxiety, embarrassment, frustration, relief, validation), "
    "actions/behaviors (massage, breathing techniques, carbonation tests, dietary changes, booking appointments), "
    "logistics (insurance, referrals, clinic names), and community/platform references (subreddit, reddit, tiktok, instagram), "
    "abbreviations of common medical terms, and reasonable umbrella/hypernym lay terms (e.g., heart_condition, throat_condition)."
)

SYSTEM_PROMPT = (
    "You are a semantic similarity decider for short terms and underscore-separated MWEs.\n\n"
    f"DOMAIN CONTEXT:\n{RCPD_DOMAIN_CONTEXT}\n\n"
    "TASK (STRICT): Reply Y to accept (hypernym/hyponym/holonym/cohyponym/synonym/near-syn/morph variant in this domain). "
    "Reply N to reject. Do not explain. Reply only with Y or N."
)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Interactive Y/N decider with probabilities (llama-server)")
    ap.add_argument("--url", type=str, default="http://localhost:18080/v1/chat/completions",
                    help="llama-server chat completions endpoint")
    # --model is now optional; we can autodetect from /v1/models
    ap.add_argument("--model", type=str, default=None,
                    help="Model ID as shown by /v1/models (often the full GGUF path). If omitted, auto-detect the first available model.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    ap.add_argument("--top-k", dest="top_k", type=int, default=0)
    ap.add_argument("--top-logprobs", dest="top_logprobs", type=int, default=10,
                    help="How many alternatives to return per token (raise to ensure both Y and N appear)")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON answer/probs")
    return ap.parse_args()

def _models_url_from_chat_url(chat_url: str) -> str:
    """
    Build /v1/models URL from a given chat completions URL.
    - If the path ends with /v1/chat/completions, replace with /v1/models.
    - Otherwise, keep scheme/host/port and set path to /v1/models.
    """
    p = urlparse(chat_url)
    path = p.path.rstrip("/")
    if path.endswith("/v1/chat/completions"):
        new_path = path[:-len("/v1/chat/completions")] + "/v1/models"
    else:
        new_path = "/v1/models"
    return urlunparse((p.scheme, p.netloc, new_path, "", "", ""))

def resolve_model_id(session: requests.Session, chat_url: str, provided: Optional[str], timeout: int = 10) -> str:
    """
    If provided is not None, return it.
    Else call /v1/models and pick the first model ID (OpenAI-style 'data[0].id' if present,
    otherwise try 'models[0].id/model/name').
    """
    if provided:
        return provided
    models_url = _models_url_from_chat_url(chat_url)
    r = session.get(models_url, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    # Prefer OpenAI-style data list
    if isinstance(data, dict) and "data" in data and data["data"]:
        first = data["data"][0]
        # typical key is 'id'
        mid = first.get("id")
        if mid:
            return mid

    # Fallback to llama-server "models" list
    if isinstance(data, dict) and "models" in data and data["models"]:
        m = data["models"][0]
        for key in ("id", "model", "name"):
            if key in m and m[key]:
                return m[key]

    raise RuntimeError("Could not auto-detect a model from /v1/models")

def build_payload(model: str, seed: str, candidate: str,
                  temperature: float, top_p: float, top_k: int, top_logprobs: int) -> Dict[str, Any]:
    return {
        "model": model,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "logprobs": True,
        "top_logprobs": int(top_logprobs),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"SEED: {seed}\nCANDIDATE: {candidate}"}
        ]
    }

def extract_answer(content_str: str) -> Optional[str]:
    """
    The server sends choices[0].message.content as a JSON string ("\"Y\"" or "\"N\"")
    when using a JSON-schema that enforces a string. Try to parse; fallback to raw.
    """
    try:
        val = json.loads(content_str)
        if isinstance(val, str) and val in ("Y", "N"):
            return val
    except Exception:
        pass
    s = content_str.strip().strip('"').strip()
    if s in ("Y", "N"):
        return s
    return None

def extract_probs(logprobs_block: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    From OpenAI-style logprobs:
      logprobs: { "content": [ { "token": ..., "logprob": ..., "top_logprobs": [ {...}, ... ] }, ... ] }
    We find the first step whose top_logprobs includes 'Y' or 'N' (the letter token)
    and return normalized p(Y), p(N) computed from exp(logprob).
    """
    content_steps = logprobs_block.get("content") or []
    for step in content_steps:
        top = step.get("top_logprobs") or []
        raw = {}
        for alt in top:
            tok = alt.get("token")
            lp = alt.get("logprob")
            if tok in ("Y", "N") and isinstance(lp, (float, int)):
                raw[tok] = math.exp(lp)
        if raw:
            total = sum(raw.values()) or 1.0
            pY = raw.get("Y", 0.0) / total
            pN = raw.get("N", 0.0) / total
            return pY, pN
    return None, None

def query_once(session: requests.Session, url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    r = session.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=timeout)
    r.raise_for_status()
    return r.json()

def main():
    args = parse_args()
    session = requests.Session()

    # Resolve model automatically if not provided
    try:
        model_id = resolve_model_id(session, args.url, args.model, args.timeout)
    except Exception as e:
        print(f"Error resolving model id: {e}", file=sys.stderr)
        sys.exit(1)

    print("Interactive mode. Type /q to quit, press Enter on seed> to reuse the previous seed.")
    last_seed: Optional[str] = None

    while True:
        try:
            seed = input("seed> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye!")
            return

        if seed == "/q":
            print("bye!")
            return

        if not seed:
            if last_seed is None:
                continue
            seed = last_seed
            print(f"(reusing seed: {seed})")

        try:
            candidate = input("candidate> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye!")
            return
        if candidate == "/q":
            print("bye!")
            return
        if not candidate:
            continue

        last_seed = seed

        payload = build_payload(
            model=model_id,
            seed=seed,
            candidate=candidate,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            top_logprobs=args.top_logprobs,
        )

        try:
            resp = query_once(session, args.url, payload, args.timeout)
        except Exception as e:
            print(f"HTTP error: {e}")
            continue

        # Extract answer
        answer = "?"
        try:
            choice = (resp.get("choices") or [])[0]
            message = choice.get("message") or {}
            content_str = message.get("content", "")
            answer = extract_answer(content_str) or "?"
        except Exception:
            pass

        # Extract probabilities
        pY = pN = None
        try:
            logprobs = (resp.get("choices") or [])[0].get("logprobs") or {}
            pY, pN = extract_probs(logprobs)
        except Exception:
            pass

        # Fallback distribution if we at least have an answer
        if (pY is None or pN is None) and answer in ("Y", "N"):
            if answer == "Y":
                pY, pN = 1.0, 0.0
            else:
                pY, pN = 0.0, 1.0

        if args.pretty:
            out = {
                "seed": seed,
                "candidate": candidate,
                "model": model_id,
                "answer": answer,
                "probs": {"pY": pY, "pN": pN},
            }
            print(json.dumps(out, indent=2))
        else:
            print(f"answer={answer} pY={pY} pN={pN}")

if __name__ == "__main__":
    main()
