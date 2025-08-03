#!/usr/bin/env python3
import json
import requests
import argparse
from pathlib import Path

class LemmaValidator:
    def __init__(self, model: str, url: str = "http://localhost:11434/api/chat"):
        self.url = url
        self.headers = {"Content-Type": "application/json"}
        self.default_data = {
            "model": model,
            "options": {"temperature": 0.0},
            "stream": False,
            "format": {
                "type": "object",
                "properties": {
                    "agree":        {"type": "boolean"},
                    "correct_lemma":{"type": ["string", "null"]}
                },
                "required": ["agree", "correct_lemma"]
            }
        }
        self.system_message = (
            "You are a lemmatization validator. "
            "When given a word and its proposed lemma, respond **only** in JSON with two fields: "
            "\"agree\" (true/false) and \"correct_lemma\" (the proper lemma, or null if unsure)."
        )

    def validate(self, term: str, lemma: str) -> dict:
        # 1. Build the user prompt
        user_prompt = (
            f"Word: \"{term}\"\n"
            f"Proposed lemma: \"{lemma}\"\n\n"
            "If the proposed lemma is correct, return:\n"
            "{ \"agree\": true, \"correct_lemma\": null }\n\n"
            "If incorrect, return:\n"
            "{ \"agree\": false, \"correct_lemma\": \"<correct lemma>\" }\n\n"
            "If unsure, return:\n"
            "{ \"agree\": false, \"correct_lemma\": null }\n"
        )

        # 2. Create the payload
        payload = self.default_data.copy()
        payload["messages"] = [
            {"role": "system",  "content": self.system_message},
            {"role": "user",    "content": user_prompt}
        ]

        # 3. Send the request (with a timeout so it won't hang forever)
        resp = requests.post(self.url, headers=self.headers, json=payload, timeout=30)
        resp.raise_for_status()
        body = resp.json()

        # 4. Unwrap any legacy "choices" container
        if "choices" in body:
            content = body["choices"][0]["message"]["content"]
            return json.loads(content)

        # 5. Unwrap the new top-level "message" container
        if "message" in body and isinstance(body["message"], dict):
            content = body["message"].get("content", "")
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass

        # 6. Otherwise assume it's already the structured JSON
        return body

def append_to_json(path: Path, key: str, value):
    """Load existing JSON (or start new), append key/value, and write back."""
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {}
    data[key] = value
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main(lookup_path: Path, agree_out: Path, flagged_out: Path, model: str):
    # Load your lemma_lookup.json
    data = json.loads(lookup_path.read_text(encoding="utf-8"))

    validator = LemmaValidator(model)
    agree_count = 0
    flag_count = 0

    for term, lemma in data.items():
        try:
            result = validator.validate(term, lemma)
        except Exception as e:
            print(f"⚠️ Error validating '{term}': {e}")
            result = {"agree": False, "correct_lemma": None}

        if result.get("agree"):
            append_to_json(agree_out, term, lemma)
            agree_count += 1
        else:
            new_lem = result.get("correct_lemma") or term
            append_to_json(flagged_out, term, new_lem)
            flag_count += 1

        print(f"Processed '{term}': agree={result.get('agree')}")

    print(f"✔️ Done: {agree_count} agreed, {flag_count} flagged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate lemmas via Ollama and append in real-time")
    parser.add_argument(
        "--lookup-json", "-l",
        type=Path, required=True,
        help="Path to your lemma_lookup.json file"
    )
    parser.add_argument(
        "--agree-out", "-a",
        type=Path, default=Path("lemmas_agree.json"),
        help="Output JSON for agreed entries"
    )
    parser.add_argument(
        "--flagged-out", "-f",
        type=Path, default=Path("lemmas_flagged.json"),
        help="Output JSON for corrected/unsure entries"
    )
    parser.add_argument(
        "--model", "-m",
        type=str, default="llama3.3:latest",
        help="Ollama model name (e.g. llama3.3:latest)"
    )
    args = parser.parse_args()

    main(
        lookup_path=args.lookup_json,
        agree_out=args.agree_out,
        flagged_out=args.flagged_out,
        model=args.model
    )
