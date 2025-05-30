import os
import csv
import sys
from datetime import datetime
from datasets import Dataset
import pandas as pd
import torch
import plotly.graph_objects as go
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    RobertaTokenizerFast
)

sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize

# List of (short name, model checkpoint)
MODELS = [
    # ("bert-base",      "bert-base-uncased"),
    ("bertweet",       "vinai/bertweet-base"),
    ("clinical-bert",  "emilyalsentzer/Bio_ClinicalBERT"),
    ("pubmed-bert",    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"),
]

def prepare_dataset(cleaned_docs: list[list[str]]) -> Dataset:
    texts = [" ".join(tokens) for tokens in cleaned_docs]
    return Dataset.from_dict({"text": texts})


def fine_tune_mlm(
    model_name: str,
    dataset: Dataset,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 16,
    max_length: int = 512,
    mlm_probability: float = 0.15
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model     = AutoModelForMaskedLM.from_pretrained(model_name)

    # Ensure mask and pad tokens
    special_tokens = {}
    if tokenizer.mask_token is None:
        special_tokens["mask_token"] = tokenizer.mask_token or "<mask>"
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = tokenizer.eos_token or "<pad>"
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))

    # Add corpus tokens safely for different tokenizers
    unique_tokens = set(tok for text in dataset["text"] for tok in text.split())
    new_tokens = [tok for tok in unique_tokens if tok not in tokenizer.get_vocab()]
    if new_tokens:
        # Detect Roberta-based tokenizers automatically
        if isinstance(tokenizer, RobertaTokenizerFast):
            tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        else:
            tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)

    # Tokenize and split
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True
        )

    splits   = dataset.train_test_split(test_size=0.1)
    train_ds = splits["train"].map(tokenize_fn, batched=True, remove_columns=["text"])
    eval_ds  = splits["test"].map(tokenize_fn, batched=True, remove_columns=["text"])

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )

    # Training args
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        fp16=True,
        dataloader_num_workers=4,
        learning_rate=3e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=2,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    # Train + save
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save eval loss history
    log_history  = trainer.state.log_history
    eval_entries = [e for e in log_history if "eval_loss" in e]
    csv_path     = os.path.join(output_dir, "eval_loss.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "eval_loss"])
        for entry in eval_entries:
            writer.writerow([entry.get("epoch"), entry.get("eval_loss")])

if __name__ == "__main__":
    raw = return_documents("reddit", "noburp_all", ["noburp"])
    texts = [" ".join(doc) if isinstance(doc, list) else str(doc) for doc in raw if doc]
    cleaned = [clean_and_tokenize(t) for t in texts]
    ds = prepare_dataset(cleaned)

    timestamp = datetime.now().strftime("%m_%d_%H_%M")
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"finetuned_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)

    for short_name, checkpoint in MODELS:
        outdir = os.path.join(base_dir, short_name)
        os.makedirs(outdir, exist_ok=True)
        print(f"Fine-tuning {checkpoint} → {outdir}")
        fine_tune_mlm(checkpoint, ds, outdir)
