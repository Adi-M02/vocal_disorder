import sys

sys.path.append('../vocal_disorder')

from utils.text_pipeline import process_text

output_path = "testing/test_manual_terms.txt"

if __name__ == "__main__":
    # Read all terms and create a set for uniqueness
    unique_terms = set()
    with open("vocabulary_evaluation/manual_terms_7_12/manual_terms.txt") as f:
        for line in f:
            terms = [term.strip() for term in line.split(',') if term.strip()]
            unique_terms.update(terms)

    with open(output_path, "w") as out_f:
        for term in unique_terms:
            processed = process_text(term, stoplist=False)
            # If processed is a list, join with space
            if isinstance(processed, (list, tuple)):
                processed_str = ' '.join(str(x) for x in processed)
            else:
                processed_str = str(processed)
            out_f.write(f"{term} -> {processed_str}\n")