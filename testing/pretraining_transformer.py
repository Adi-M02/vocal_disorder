"""
Continued-pretraining Bio-ClinicalBERT on r/noburp posts and comments
────────────────────────────────────────────────────────
* Implements:
  • extended vocab for domain terms
  • sliding-window tokenisation (512 / 128 stride)
  • dynamic padding + whole-word masking
  • perplexity metric shown during eval
"""

import sys, os, re, html, math
import torch
import initial_analyses as ia                      
sys.path.append("/local/disk2/not_backed_up/amukundan/research/vocal_disorder")
import query_mongo as query                          # local helper

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForWholeWordMask,
    Trainer,
    TrainingArguments,
)

# ────────────────────────────────────────────────────
# 0. Static config
# ────────────────────────────────────────────────────
MAIN_RCPD_SUBREDDITS = ["noburp"]
MAX_SEQ_LEN         = 512          # BioClinicalBERT’s maximum
SPAN_STRIDE         = 128          # overlap between windows
MLM_PROB            = 0.15

DOMAIN_TERMS = [
    "rcpd", "noburp", "botox", "cricopharyngeal", "myotomy",
    "aerophagia", "gastroparesis", "ent",
    "gastroenterologist", "gastroenterology", "emetophobe"
]

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# ────────────────────────────────────────────────────
# 1. Tokenizer & model (with vocab extension)
# ────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

new_tokens = [t for t in DOMAIN_TERMS if len(tokenizer.tokenize(t)) > 1]
if new_tokens:
    tokenizer.add_tokens(new_tokens)
    print(f"Added {len(new_tokens)} domain tokens → new vocab size = {len(tokenizer)}")

model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
if new_tokens:
    model.resize_token_embeddings(len(tokenizer))

# ────────────────────────────────────────────────────
# 2. Mongo → raw text
# ────────────────────────────────────────────────────
def get_noburp_text():
    """
    Returns a list of cleaned self-texts from r/noburp posts and comments.
    """
    raw_texts = []
    for text in query.get_text_by_subreddits(MAIN_RCPD_SUBREDDITS):
        text = re.sub(r"\s+", " ", text)
        raw_texts.append(text)
    return raw_texts

# ────────────────────────────────────────────────────
# 3. Tokenisation helper
# ────────────────────────────────────────────────────
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        stride=SPAN_STRIDE,
        return_overflowing_tokens=True,
        return_special_tokens_mask=True,   # for WWM collator
    )

# ────────────────────────────────────────────────────
# 4. Metric: perplexity
# ────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    """Hugging-Face expects a dict."""
    loss = eval_pred[0]
    return {"perplexity": math.exp(loss)}

# ────────────────────────────────────────────────────
# 5. Main training routine
# ────────────────────────────────────────────────────
def main():

    # 5.1 Load data
    raw_posts = get_noburp_text()
    if not raw_posts:
        print("No NoBurp posts found in the database.")
        return
    print(f"Loaded {len(raw_posts)} posts.")

    # 5.2 Build Dataset
    dataset = Dataset.from_dict({"text": raw_posts})

    # 5.3 Tokenise with sliding windows
    tokenized_dataset = (
        dataset
        .map(tokenize_function, batched=True, remove_columns=["text"])
        .remove_columns(["overflow_to_sample_mapping"])   # keep special_tokens_mask
    )

    # 5.4 Train/val split
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)} examples,  Eval: {len(eval_ds)} examples")

    # 5.5 Data collator (dynamic padding + whole-word mask)
    data_collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer,
        mlm_probability=MLM_PROB,
    )
    # 5.6 training args (no eval, OOM issue)
    training_args = TrainingArguments(
        output_dir="bioclinicalbert_noburp_all_pretrained",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        logging_steps=1000,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    # 5.7 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # 5.8 Train
    checkpoint_dir = training_args.output_dir
    last_checkpoint = None
    if os.path.isdir(checkpoint_dir):
        checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=os.path.getctime)
    train_output = trainer.train(resume_from_checkpoint=last_checkpoint)

    # 5.9 Save final model & tokenizer
    model_dir = "bioclinicalbert_noburp_all/model"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(os.path.join(model_dir, "tokenizer"))

    # 5.10 Save training metrics
    metrics_path = os.path.join("bioclinicalbert_noburp_all", "train_metrics.log")
    with open(metrics_path, "w", encoding="utf-8") as f:
        import json
        json.dump(train_output.metrics, f, indent=4)
    print(f"✅  Pretraining completed. Model saved to {model_dir}")
    print(f"📊  Training metrics saved to {metrics_path}")
# ────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
