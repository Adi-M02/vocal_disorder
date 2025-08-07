import sys
import re
from langdetect import detect_langs, DetectorFactory, LangDetectException

# deterministic results
DetectorFactory.seed = 0

sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize

def normalize(text):
    text = re.sub(r'https?://\S+|www\.\S+|[\w\.-]+@[\w\.-]+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def write_non_english_report(min_confidence=0.95, text_output_path='non_english.txt'):
    non_eng = []
    docs = return_documents(
        db_name="reddit",
        collection_name="noburp_all",
        filter_subreddits=["noburp"],
        mongo_uri="mongodb://localhost:27017/",
    )
    for idx, doc in enumerate(docs, 1):
        raw       = doc
        normalized= normalize(raw)  # or normalize(cleaned)
        if len(normalized) < 30:
            continue

        try:
            langs = detect_langs(normalized)
            if not langs:
                continue

            top = langs[0]
            is_non_en = top.lang != 'en' and top.prob >= min_confidence

            # if there *is* a runner-up, also demand a gap
            if len(langs) >= 2:
                second = langs[1]
                is_non_en = is_non_en and (top.prob - second.prob) >= 0.15

            if is_non_en:
                non_eng.append({
                    "doc_num":   idx,
                    "lang":      top.lang,
                    "prob":      float(top.prob),
                    "full_text": raw.replace('\n', ' ')
                })
        except LangDetectException:
            continue

    # 2) Write the report
    total_checked = len(docs)
    total_flagged = len(non_eng)
    with open(text_output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("Non-English Posts Report\n")
        f.write("=========================\n")
        f.write(f"Total documents checked: {total_checked}\n")
        f.write(f"Non-English (≥{min_confidence*100:.0f}% conf): {total_flagged}\n\n")

        # Each document block
        for entry in non_eng:
            f.write(f"----- Document {entry['doc_num']} of {total_flagged} -----\n")
            f.write(f"Language   : {entry['lang']}  (confidence {entry['prob']*100:.1f}%)\n")
            f.write("Full text  :\n")
            f.write(entry['full_text'] + "\n\n")

        # Footer
        f.write(f"End of report — {total_flagged} non-English documents listed.\n")

    print(f"Report written to {text_output_path} — {total_flagged} non-English posts found.")
    return non_eng

if __name__ == "__main__":
    write_non_english_report(min_confidence=0.85,
                             text_output_path='testing/language_detection/non_english_docs.txt')
