#!/usr/bin/env python3
"""
Interactive seed+candidate CLI — llama.cpp only
- STRICT JSON (stdout) via /v1/chat/completions + response_format schema
- Y/N n_probs (stderr) via /completion + grammar (pre-sampling probs)
"""

from __future__ import annotations
import argparse, json, math, sys, threading, time
from pathlib import Path
from typing import List, Optional, Dict, Any
import requests

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

RELATION_ENUM = [
    "hypernym","hyponym","holonym","cohyponym",
    "synonym","morphological_variant","near_synonym",
    "unrelated","unknown",
]

QUIT_TOKENS = {"/q","/quit","quit","exit",":q",":qa"}

class Spinner:
    FRAMES = ["|","/","-","\\"]
    def __init__(self, text=" contacting LLM… ", enabled=True, interval=0.12):
        self.text, self.enabled, self.interval = text, enabled, interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
    def start(self):
        if not self.enabled or self._thread: return
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
        sys.stdout.write("\r" + " " * (len(self.text)+4) + "\r")
        sys.stdout.flush()
    def stop(self):
        if not self._thread: return
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None

def build_system_closure(ctx: str, include_relation: bool) -> str:
    return (
        "You are expanding a concept bucket defined by the FULL SET OF SEED TERMS (potentially ~2000 strings).\n"
        "You will be given:\n"
        " • SEED (the focal term),\n"
        " • VOCAB_BUCKET_SEEDS (a long list of seed terms acting as anchors/context), and\n"
        " • EXACTLY ONE REMAINING_CANDIDATE.\n"
        "Return STRICT JSON {seed, decisions"
        + (", relation" if include_relation else "")
        + "}.\n"
        "The ONLY valid outputs for 'decisions' are [] or [<exact candidate>].\n"
        "Apply the same acceptance rules (hypernym, hyponym, holonym, cohyponym, synonyms, morphology).\n\n"
        f"DOMAIN CONTEXT:\n{ctx}\n"
    )

def build_user_closure(seed: str, anchors: List[str], candidate: str, include_relation: bool) -> str:
    anc_str = "\n".join(f"- {a}" for a in anchors) if anchors else "(none)"
    base = (
        f"SEED: {seed}\n"
        f"VOCAB_BUCKET_SEEDS (long list acting as anchors/context):\n{anc_str}\n\n"
        f"REMAINING_CANDIDATE (evaluate ONLY this exact string):\n- {candidate}\n\n"
    )
    return base + (
        'Respond ONLY with JSON {"seed": <seed>, "decisions": [] or [<exact candidate>], optionally "relation": <one label>}.'
        if include_relation else
        'Respond ONLY with JSON {"seed": <seed>, "decisions": [] or [<exact candidate>]}.'
    )

def build_single_candidate_schema(candidate: str, include_relation: bool) -> dict:
    props = {
        "seed": {"type": "string"},
        "decisions": {
            "oneOf": [
                {"type": "array", "maxItems": 0},
                {"type": "array", "minItems": 1, "maxItems": 1, "items": {"const": candidate}},
            ]
        },
    }
    required = ["seed", "decisions"]
    if include_relation:
        props["relation"] = {"type": "string", "enum": RELATION_ENUM}
    return {"type": "object","additionalProperties": False,"required": required,"properties": props}

class LlamaCppEngine:
    """llama.cpp only: strict JSON via /v1/chat/completions + n_probs via /completion"""
    def __init__(self, model: str, base_url: str, temperature: float, max_new_tokens: int,
                 timeout: int, include_relation: bool, global_context: str,
                 nprobs_k: int = 5, session: Optional[requests.Session] = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.timeout = timeout
        self.include_relation = include_relation
        self.global_context = (global_context or GLOBAL_CONTEXT_DEFAULT).strip()
        self.nprobs_k = max(1, int(nprobs_k))
        self.session = session or requests.Session()
        self.system_closure = build_system_closure(self.global_context, include_relation)

    # Strict JSON (stdout) — always using closure + anchors
    def _chat_schema(self, seed: str, anchors: List[str], candidate: str) -> dict:
        schema = build_single_candidate_schema(candidate, self.include_relation)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_closure},
                {"role": "user", "content": build_user_closure(seed, anchors, candidate, self.include_relation)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "response_format": {"type": "json_object", "schema": schema},
            "cache_prompt": True
        }
        r = self.session.post(f"{self.base_url}/v1/chat/completions",
                              headers={"Content-Type": "application/json"},
                              json=payload, timeout=(10, self.timeout))
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)

    def call_with_anchors_raw(self, seed: str, anchors: List[str], candidate: str) -> dict:
        return self._chat_schema(seed, anchors, candidate)

    # n_probs (stderr) — pre-sampling, aligned with closure prompt
    def get_nprobs(self, seed: str, candidate: str, anchors: List[str]) -> Dict[str, Any]:
        def _canon(tok: str) -> str:
            return (tok or "").replace("▁", " ").strip()

        anc_block = "VOCAB_BUCKET_SEEDS (anchors):\n" + "\n".join(f"- {a}" for a in anchors) + "\n\n"
        prompt = (
            f"{self.system_closure}\n\n"
            f"SEED: {seed}\n{anc_block}"
            f"REMAINING_CANDIDATE: {candidate}\n\n"
            "Answer exactly one character: Y (ACCEPT) or N (REJECT) for the candidate in this R-CPD context."
        )

        sse_payload = {
            "prompt": prompt,
            "n_predict": 1,
            "temperature": 0.0,
            "grammar": 'root ::= "Y" | "N"',
            "n_probs": self.nprobs_k,
            "top_k": 2,
            "post_sampling_probs": False,     # << pre-sampling for meaningful Y/N
            "stream": True,
            "cache_prompt": True,
        }

        choice, merged, resp = "", {}, None
        try:
            resp = self.session.post(f"{self.base_url}/completion",
                                     headers={"Content-Type": "application/json"},
                                     json=sse_payload, timeout=(10, self.timeout), stream=True)
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                try:
                    evt = json.loads(raw[6:])
                except Exception:
                    continue
                if not choice:
                    choice = (evt.get("content") or "").strip()[:1]
                cps = evt.get("completion_probabilities")
                if isinstance(cps, list) and cps:
                    probs = cps[0].get("probs") or cps[0].get("top_logprobs") or []
                    for it in probs:
                        tok = _canon(it.get("tok_str") or it.get("token") or "")
                        p = it.get("prob")
                        if p is None and "logprob" in it:
                            try: p = math.exp(float(it["logprob"]))
                            except Exception: p = None
                        if tok and isinstance(p, (int, float)):
                            merged[tok] = max(float(p), merged.get(tok, 0.0))
                    break
                if evt.get("stop") is True:
                    break
        finally:
            if resp is not None:
                try: resp.close()
                except Exception: pass

        # Fallback: non-stream final JSON
        if not merged:
            ns_payload = dict(sse_payload); ns_payload["stream"] = False
            r2 = self.session.post(f"{self.base_url}/completion",
                                   headers={"Content-Type": "application/json"},
                                   json=ns_payload, timeout=(10, self.timeout))
            r2.raise_for_status()
            j2 = r2.json()
            choice = (choice or (j2.get("content","") or j2.get("completion",""))).strip()[:1]
            cp = j2.get("completion_probabilities")
            if isinstance(cp, list) and cp:
                probs = cp[0].get("probs") or cp[0].get("top_logprobs") or []
                for it in probs:
                    tok = _canon(it.get("tok_str") or it.get("token") or "")
                    p = it.get("prob")
                    if p is None and "logprob" in it:
                        try: p = math.exp(float(it["logprob"]))
                        except Exception: p = None
                    if tok and isinstance(p, (int, float)):
                        merged[tok] = max(float(p), merged.get(tok, 0.0))
            toks = j2.get("tokens")
            if isinstance(toks, list) and toks:
                t0 = toks[0]
                for it in (t0.get("top_logprobs") or []):
                    tok = _canon(it.get("token") or it.get("tok_str") or "")
                    p = it.get("prob")
                    if p is None and "logprob" in it:
                        try: p = math.exp(float(it["logprob"]))
                        except Exception: p = None
                    if tok and isinstance(p, (int, float)):
                        merged[tok] = max(float(p), merged.get(tok, 0.0))

        # Ensure both Y and N are present (since grammar is {Y|N})
        pY, pN = merged.get("Y"), merged.get("N")
        if pY is not None and pN is None:
            merged["N"] = max(0.0, 1.0 - pY); pN = merged["N"]
        elif pN is not None and pY is None:
            merged["Y"] = max(0.0, 1.0 - pN); pY = merged["Y"]
        elif pY is None and pN is None and choice in ("Y","N"):
            # worst-case: no probs returned; infer a degenerate distribution
            merged[choice] = 1.0
            if choice == "Y": merged["N"] = 0.0
            else:             merged["Y"] = 0.0
            pY, pN = merged["Y"], merged["N"]

        denom = (pY or 0.0) + (pN or 0.0) or 1.0
        confidence = (pY or 0.0) / denom

        probs_list = [{"token": k, "prob": v} for k, v in sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[: self.nprobs_k]]
        return {"choice": choice if choice in ("Y","N") else "", "probs": probs_list, "confidence": confidence}

def _load_anchors(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, dict): return [str(k) for k in data.keys()]
        if isinstance(data, list): return [str(t).strip() for t in data if isinstance(t, str) and str(t).strip()]
    except Exception:
        pass
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]

def interactive_loop(engine: LlamaCppEngine, anchors: List[str], pretty: bool, spinner_enabled: bool):
    print("Interactive mode. Type /help for help, /quit to exit.")
    last_seed: Optional[str] = None
    while True:
        try:
            seed = input("seed> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye!"); return
        if not seed and last_seed:
            seed = last_seed; print(f"(reusing seed: {seed})")
        elif not seed:
            continue
        if seed.lower() in QUIT_TOKENS: print("bye!"); return
        if seed.lower() in {"/help","help","h","?"}:
            print("Commands: /q, /quit, quit, exit ; Enter on seed> reuses previous seed"); continue

        try:
            candidate = input("candidate> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye!"); return
        if candidate.lower() in QUIT_TOKENS: print("bye!"); return
        if not candidate: continue
        last_seed = seed

        spin = Spinner(enabled=spinner_enabled)
        try:
            spin.start()
            out = engine.call_with_anchors_raw(seed, anchors, candidate)
        except Exception as e:
            spin.stop()
            print(json.dumps({"error": str(e)}, ensure_ascii=False))
            continue
        finally:
            spin.stop()

        if pretty: print(json.dumps(out, indent=2, ensure_ascii=False))
        else:      print(json.dumps(out, ensure_ascii=False))

        # Y/N n_probs (stderr)
        try:
            np_out = engine.get_nprobs(seed=seed, candidate=candidate, anchors=anchors)
            probs_str = ", ".join(f"{p['token']}={p['prob']:.4f}" for p in np_out.get("probs", [])[:5])
            sys.stderr.write(f"[n_probs] choice={np_out.get('choice','')} conf={np_out.get('confidence',0.0):.4f} :: {probs_str}\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[n_probs] error: {e}\n"); sys.stderr.flush()

def main():
    ap = argparse.ArgumentParser(description="Interactive strict single-term decision via llama.cpp (stdout JSON; stderr n_probs).")
    ap.add_argument("--model", type=str, default="model", help="Model name label (llama.cpp may ignore if a single model is loaded)")
    ap.add_argument("--url",   type=str, default="http://localhost:8080", help="llama.cpp server base URL (no path)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tokens", type=int, default=128, help="max new tokens for chat completion")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--anchors", type=str, default="testing/llm_addition_testing/8-24/single_with_anchor/single_term_eval_08_24_22_57/accepted_aligned_by_seed.json", help="JSON object(keys)/array OR TXT (one per line)")
    ap.add_argument("--global_context", type=str, default=GLOBAL_CONTEXT_DEFAULT)
    ap.add_argument("--pretty", action="store_true", default=True)
    ap.add_argument("--no-spinner", action="store_true")
    ap.add_argument("--nprobs-k", type=int, default=5)
    args = ap.parse_args()

    session = requests.Session()
    engine = LlamaCppEngine(
        model=args.model,
        base_url=args.url,
        temperature=args.temperature,
        max_new_tokens=args.tokens,
        timeout=args.timeout,
        include_relation=True,
        global_context=args.global_context,
        nprobs_k=args.nprobs_k,
        session=session,
    )

    p = Path(args.anchors)
    if not p.exists():
        sys.stderr.write(f"[error] anchors file not found: {p}\n")
        sys.exit(2)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            anchors = [k for k, v in data.items() if isinstance(v, list) and len(v) > 0]
            if not anchors:
                anchors = list(data.keys())
        else:
            anchors = _load_anchors(p)
    except Exception:
        anchors = _load_anchors(p)

    try:
        interactive_loop(engine, anchors, bool(args.pretty), not args.no_spinner)
    except KeyboardInterrupt:
        print("\nbye!")

if __name__ == "__main__":
    main()
