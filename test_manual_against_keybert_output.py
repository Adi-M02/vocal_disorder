import json
from pathlib import Path
from typing import List, Tuple
import re
from itertools import chain

def check_terms_against_keybert_output(
    json_path: str,
    terms_txt_path: str,
    write_files: bool = False,
    out_prefix: str = None
):
    """
    Loads your JSON and comma-separated TXT, then prints which terms are present
    in json['phrase_counts'] and which are not. If write_files=True, also writes:
      - {out_prefix}.json     (summary)
      - {out_prefix}_found.txt
      - {out_prefix}_not_found.txt
    """
    # Resolve paths
    json_path = Path(json_path)
    terms_txt = Path(terms_txt_path)
    if out_prefix:
        prefix = Path(out_prefix).stem
    else:
        prefix = json_path.stem + "_term_check"

    # Load phrase keys
    data = json.loads(json_path.read_text(encoding="utf-8"))
    phrase_keys = set(data.get("phrase_counts", {}).keys())

    # Load and split comma-separated terms
    raw = terms_txt.read_text(encoding="utf-8").lower()
    terms = [t.strip().lower() for t in raw.split(",") if t.strip()]

    # Compare
    found     = [t for t in terms if t in phrase_keys]
    not_found = [t for t in terms if t not in phrase_keys]

    # Print results
    print(f"🔎 Checked {len(terms)} terms against {len(phrase_keys)} phrase-keys")
    print(f"\n✅ Found ({len(found)}):\n" + ", ".join(found) or "  (none)")
    print(f"\n❌ Not Found ({len(not_found)}):\n" + ", ".join(not_found) or "  (none)")

    # # Optionally write out files
    # if write_files:
    #     summary = {
    #         "checked": len(terms),
    #         "found_count": len(found),
    #         "not_found_count": len(not_found),
    #         "found": found,
    #         "not_found": not_found
    #     }
    #     json_out = json_path.with_name(f"{prefix}.json")
    #     json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    #     json_path.with_name(f"{prefix}_found.txt").write_text("\n".join(found), encoding="utf-8")
    #     json_path.with_name(f"{prefix}_not_found.txt").write_text("\n".join(not_found), encoding="utf-8")
    #     print(f"\n📝 Wrote summary to {json_out} and lists to *_found.txt / *_not_found.txt")

    return found, not_found

def check_terms_in_text_file(
    text_file: str,
    terms_txt_file: str
) -> Tuple[List[str], List[str]]:
    """
    Reads the entire contents of text_file, then reads comma-separated terms
    from terms_txt_file. Returns two lists: (found, not_found), where `found`
    are the terms that appear as whole words in text_file, and `not_found` are
    the remainder.
    """
    # Load and lowercase the corpus text
    corpus = Path(text_file).read_text(encoding="utf-8").lower()

    # Load and split comma-separated terms, strip whitespace
    raw = Path(terms_txt_file).read_text(encoding="utf-8")
    terms = [t.strip() for t in raw.split(",") if t.strip()]
    terms_lower = [t.lower() for t in terms]

    found = []
    not_found = []
    for orig, lower in zip(terms, terms_lower):
        # \b ensures word-boundary matching
        if re.search(rf'\b{re.escape(lower)}\b', corpus):
            found.append(orig)
        else:
            not_found.append(orig)

    # Print results
    print(f"🔎 Checked {len(terms)} terms against text in {text_file}")
    print(f"\n✅ Found ({len(found)}):\n  " + ", ".join(found) if found else "\n✅ Found: (none)")
    print(f"\n❌ Not Found ({len(not_found)}):\n  " + ", ".join(not_found) if not_found else "\n❌ Not Found: (none)")

    return found, not_found

def check_terms_against_vocab(
    json_path: str,
    terms_txt_path: str,
    write_files: bool = False,
    out_prefix: str = None
) -> Tuple[List[str], List[str]]:
    """
    Loads a vocabulary-structured JSON and a comma-separated TXT of terms,
    flattens all entries under json['vocabulary'], then prints which terms
    from the TXT appear in that vocabulary. Matching is done on lowercase,
    replacing underscores in vocab entries with spaces. If write_files=True,
    also writes summary and _found/_not_found files. Returns (found, not_found).
    """
    json_path = Path(json_path)
    terms_txt = Path(terms_txt_path)
    prefix = Path(out_prefix).stem if out_prefix else json_path.stem + "_vocab_check"

    # Load and flatten vocabulary entries
    data = json.loads(json_path.read_text(encoding="utf-8"))
    vocab_raw = list(chain.from_iterable(data.get("vocabulary", {}).values()))
    # Normalize vocab terms: lowercase and replace underscores
    vocab_set = {t.lower().replace('_', ' ') for t in vocab_raw}

    # Load and split comma-separated terms
    raw = terms_txt.read_text(encoding="utf-8").lower()
    terms = [t.strip() for t in raw.split(",") if t.strip()]

    # Compare
    found = [t for t in terms if t in vocab_set]
    not_found = [t for t in terms if t not in vocab_set]

    # Print
    print(f"🔎 Checked {len(terms)} terms against {len(vocab_set)} vocab terms")
    print(f"\n✅ Found ({len(found)}):\n" + ", ".join(found) if found else "\n✅ Found: (none)")
    print(f"\n❌ Not Found ({len(not_found)}):\n" + ", ".join(not_found) if not_found else "\n❌ Not Found: (none)")

    # Optionally write files
    if write_files:
        summary = {
            "checked": len(terms),
            "found_count": len(found),
            "not_found_count": len(not_found),
            "found": found,
            "not_found": not_found
        }
        json_out = json_path.with_name(f"{prefix}.json")
        json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        json_path.with_name(f"{prefix}_found.txt").write_text("\n".join(found), encoding="utf-8")
        json_path.with_name(f"{prefix}_not_found.txt").write_text("\n".join(not_found), encoding="utf-8")
        print(f"\n📝 Wrote summary to {json_out} and lists to *_found.txt / *_not_found.txt")

    return found, not_found
if __name__ == "__main__":
    # Example usage:
    # check_terms_against_json("my_phrases.json", "terms_to_check.txt")
    # check_terms_against_json("my_phrases.json", "terms_to_check.txt", write_files=True)
    check_terms_against_keybert_output("keybert_outputs/keybert_run_k4_ng1-3_05_08_12_16_27.json", "vocabulary_evaluation/manual_terms.txt", write_files=True, out_prefix="my_report")
    # check_terms_against_vocab("vocab_output_05_21/expanded_output_05_21_15_31_28.json", "vocabulary_evaluation/manual_terms.txt")
    # check_terms_in_text_file("user_posts_all.txt", "vocabulary_evaluation/manual_terms.txt")
# Print + write files:
# check_terms_against_json("my_phrases.json", "terms_to_check.txt", write_files=True, out_prefix="my_report")
