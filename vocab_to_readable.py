import json
from pathlib import Path

# === CONFIG ===
INPUT_JSON = Path("vocab_output_04_18/expanded_vocab_ngram_top20_uf5_bf3_tf2.json")

def format_vocab_for_word(vocab_dict):
    lines = []

    lines.append("Expanded Vocabulary by Category")
    lines.append("=" * 35)
    lines.append("")

    for category, terms in sorted(vocab_dict.items()):
        lines.append(f"{category} ({len(terms)} terms)")
        lines.append("-" * len(lines[-1]))

        sorted_terms = sorted(set(terms), key=lambda x: x.lower())
        for term in sorted_terms:
            lines.append(term)
        lines.append("")  # add spacing between sections

    return "\n".join(lines)

def main():
    with INPUT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    vocab_dict = data.get("vocabulary", {})

    formatted_output = format_vocab_for_word(vocab_dict)

    # Create output path in the same directory with modified name
    output_path = INPUT_JSON.with_name(INPUT_JSON.stem + "_word_friendly.txt")

    with output_path.open("w", encoding="utf-8") as f:
        f.write(formatted_output)

    print(f"✅ Saved Word-friendly output to: {output_path}")

if __name__ == "__main__":
    main()
