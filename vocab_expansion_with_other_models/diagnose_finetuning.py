import os
import sys
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize

def prepare_dataset(texts):
    """
    Build a HuggingFace Dataset from a list of raw text strings.
    """
    return Dataset.from_dict({"text": texts})


def main():
    # 1) Fetch & prepare
    print("Fetching documents…")
    raw = return_documents("reddit", "noburp_all", ["noburp"])
    texts = []
    # raw is a list of token lists: each doc is [term1, term2, ...]
    for doc in raw:
        if isinstance(doc, list):
            # Join tokens back into a raw text string
            text = " ".join(doc)
        else:
            text = str(doc)
        if text:
            texts.append(text)
    print(f"Total examples fetched: {len(texts)}")

    # 2) Clean & wrap in Dataset
    print("Cleaning & tokenizing…")
    cleaned = [clean_and_tokenize(t) for t in texts]
    joined = [" ".join(tokens) for tokens in cleaned]
    ds = prepare_dataset(joined)

    # 3) Train/test split
    print("Creating train/test split (90/10)…")
    splits = ds.train_test_split(test_size=0.1)
    eval_ds = splits["test"]
    print(f"Eval set size: {len(eval_ds)} examples")

    # 4) Load tokenizer for one model
    model_checkpoint = "bert-base-uncased"
    print(f"Loading tokenizer: {model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    # 5) Show raw eval examples
    print("\n--- Raw Eval Examples ---")
    for i in range(min(5, len(eval_ds))):
        ex_text = eval_ds[i]["text"]
        print(f"Example {i}: {ex_text[:100]}...")

    # 6) Tokenize eval_ds
    def tok_fn(exs):
        return tokenizer(
            exs["text"],
            truncation=True,
            max_length=128,
            return_special_tokens_mask=True
        )
    tokenized = eval_ds.map(tok_fn, batched=True, remove_columns=["text"])
    print(f"Tokenized eval features: {tokenized.column_names}")

    # 7) Inspect DataCollator masking
    print("\n--- DataCollator Masking Test ---")
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
    batch = collator([tokenized[i] for i in range(min(4, len(tokenized)))])
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    print("Batch shapes → input_ids:", input_ids.shape, ", labels:", labels.shape)
    mask_pos = (labels[0] != -100).nonzero(as_tuple=True)[0]
    print(f"Example 0 → {len(mask_pos)} masked tokens out of {labels.shape[1]}")
    for pos in mask_pos.tolist()[:10]:
        orig = tokenizer.convert_ids_to_tokens(input_ids[0][pos].item())
        lab = tokenizer.convert_ids_to_tokens(labels[0][pos].item())
        print(f"  pos {pos}: input={orig}, label={lab}")

    # 8) Small subset test
    print("\n--- Small Subset Test (100 examples) ---")
    small = ds.select(range(min(100, len(ds)))).train_test_split(test_size=0.1)["test"]
    tokenized_small = small.map(tok_fn, batched=True, remove_columns=["text"])
    batch_small = collator([tokenized_small[i] for i in range(min(4, len(tokenized_small)))])
    mask_count_small = (batch_small["labels"] != -100).sum().item()
    print(f"Total masked tokens in small batch: {mask_count_small}")

    # 9) Reduced MLM probability test
    print("\n--- Reduced MLM Probability Test (p=0.05) ---")
    collator_low = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.05)
    batch_low = collator_low([tokenized[i] for i in range(min(4, len(tokenized)))])
    mask_pos_low = (batch_low["labels"][0] != -100).nonzero(as_tuple=True)[0]
    print(f"Example 0 masked count @5%: {len(mask_pos_low)}")

if __name__ == "__main__":
    # Suppress tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
