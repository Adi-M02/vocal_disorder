import re
import argparse
from pathlib import Path

def pluralize(word: str) -> str:
    """
    Very simple English pluralizer:
     - adds “es” if word ends in s, x, z, ch, or sh
     - changes “y” → “ies” if preceded by a consonant
     - otherwise just adds “s”
    """
    if re.search(r'(s|x|z|ch|sh)$', word):
        return word + 'es'
    elif re.search(r'[^aeiou]y$', word):
        return word[:-1] + 'ies'
    else:
        return word + 's'

def ing_form(word: str) -> str:
    """
    Simplistic “-ing” form:
     - if it ends in a single ‘e’ (not “ee”), drop the ‘e’ and add “ing”
     - if consonant–vowel–consonant, double final consonant and add “ing”
     - otherwise just add “ing”
    """
    if len(word) > 1 and word.endswith('e') and not word.endswith('ee'):
        return word[:-1] + 'ing'
    elif re.search(r'[^aeiou][aeiou][^aeiou]$', word):
        return word + word[-1] + 'ing'
    else:
        return word + 'ing'

def past_form(word: str) -> str:
    """
    Simple past tense:
     - if ends in ‘e’, just add “d”
     - if CVC pattern, double final consonant and add “ed”
     - otherwise add “ed”
    """
    if word.endswith('e'):
        return word + 'd'
    elif re.search(r'[^aeiou][aeiou][^aeiou]$', word):
        return word + word[-1] + 'ed'
    else:
        return word + 'ed'

def er_form(word: str) -> str:
    """
    “-er” form (agentive/adjective):
     - if ends in ‘e’, drop that and add “er”
     - otherwise just add “er”
    """
    if word.endswith('e'):
        return word[:-1] + 'er'
    else:
        return word + 'er'


def load_and_dedup_terms(input_path: Path) -> set[str]:
    """
    Read the entire input file as text, split on commas,
    strip whitespace, lowercase, and return a set.
    """
    raw = input_path.read_text(encoding='utf-8')
    tokens = [t.strip().lower() for t in raw.split(',') if t.strip()]
    return set(tokens)


def write_variants(terms: set[str], output_path: Path) -> None:
    """
    For each term in the set, generate variants if it’s purely [a-z]+.
    Then write one comma-separated line per original term to output_path.
    """
    with output_path.open('w', encoding='utf-8') as fout:
        for term in sorted(terms):
            # If term is strictly lowercase letters, generate variants:
            if re.fullmatch(r'[a-z]+', term):
                p = pluralize(term)
                ing = ing_form(term)
                past = past_form(term)
                er_ = er_form(term)
                variants = [term, p, ing, past, er_]
            else:
                # if not purely alphabetic (e.g. “gas-x” or “marfan2”), just pass it through
                variants = [term]
            fout.write(', '.join(variants) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Read comma-separated words from INPUT_FILE, "
                    "generate basic morphological variants, "
                    "and write them (one CSV line each) to OUTPUT_FILE."
    )
    parser.add_argument(
        'input_file',
        type=Path,
        help="Path to a TXT file containing comma-separated words"
    )
    parser.add_argument(
        'output_file',
        type=Path,
        help="Path where you want the variants‐per‐line TXT file to be written"
    )
    args = parser.parse_args()

    # 1) Load & dedupe
    terms = load_and_dedup_terms(args.input_file)

    # 2) Generate variants & write
    write_variants(terms, args.output_file)

    print(f"Processed {len(terms)} unique terms and wrote variants to {args.output_file!s}")


if __name__ == "__main__":
    main()