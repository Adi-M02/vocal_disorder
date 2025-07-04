import os
import sys
from datetime import datetime

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
import plotly.graph_objects as go

sys.path.append('../vocal_disorder')
from query_mongo import return_documents

# List of (short name, model checkpoint)
MODELS = [
    ("bertweet",      "vinai/bertweet-base"),
    # ("bert-base",      "bert-base-uncased"),
    # ("clinical-bert",  "emilyalsentzer/Bio_ClinicalBERT"),
    # ("pubmed-bert",    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"),
]

def prepare_dataset(raw_texts: list[str]) -> Dataset:
    """
    Build a HuggingFace Dataset directly from raw text strings.
    """
    return Dataset.from_dict({"text": raw_texts})

def fine_tune_mlm(
    model_name: str,
    dataset: Dataset,
    output_dir: str,
    epochs: int = 15,
    batch_size: int = 16,
    mlm_probability: float = 0.15
):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # ─── 1) Load tokenizer & model ────────────────────────────────────────────
    if "bertweet" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            normalization=True,
            use_fast=False
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)

    # ─── 2) Set max_length based on model type ────────────────────────────────
    # For BERTweet and Clinical‐BERT, use 128; for Bert‐base and PubMed‐BERT, use 512.
    lc_name = model_name.lower()
    if "bertweet" in lc_name or "clinical-bert" in lc_name:
        max_length = 128
    else:
        max_length = 512

    tokenizer.model_max_length = max_length
    stride = 64  # overlap for sliding‐window

    # ─── 3) Tokenize with sliding‐window stride ────────────────────────────────
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_special_tokens_mask=True
        )

    splits = dataset.train_test_split(test_size=0.1, seed=42)
    tokenized = DatasetDict({
        "train": splits["train"].map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"]
        ),
        "eval": splits["test"].map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"]
        )
    })

    # ─── 4) ensure no token ID ≥ vocab_size ────────────────────
    for split_name in ("train", "eval"):
        for i, seq in enumerate(tokenized[split_name]["input_ids"]):
            bad_ids = [tok for tok in seq if tok >= model.config.vocab_size or tok < 0]
            if bad_ids:
                raise ValueError(
                    f"Out‐of‐range token(s) {bad_ids} in split='{split_name}', index={i}; "
                    f"vocab_size={model.config.vocab_size}"
                )

    # ─── 5) Data collator for MLM (pad to multiple of 8) ─────────────────────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
        pad_to_multiple_of=8
    )

    # ─── 6) Training arguments ─────────────────────────────────────────────────
    has_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=has_cuda,
        dataloader_num_workers=4,
        learning_rate=3e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=2,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
        seed=42
    )

    # ─── 7) Trainer setup ──────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        data_collator=data_collator,
    )

    # ─── 8) Train, then save model & tokenizer ────────────────────────────────
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ─── 9) Extract log history & plot losses ─────────────────────────────────
    log_history = trainer.state.log_history
    train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_entries  = [e for e in log_history if "eval_loss" in e]

    train_by_epoch = {entry["epoch"]: entry["loss"] for entry in train_entries}
    eval_by_epoch  = {entry["epoch"]: entry["eval_loss"] for entry in eval_entries}

    train_epochs = sorted(train_by_epoch.keys())
    eval_epochs  = sorted(eval_by_epoch.keys())

    train_losses = [train_by_epoch[ep] for ep in train_epochs]
    eval_losses  = [eval_by_epoch[ep] for ep in eval_epochs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_epochs,
        y=train_losses,
        mode="lines+markers",
        name="Training Loss"
    ))
    fig.add_trace(go.Scatter(
        x=eval_epochs,
        y=eval_losses,
        mode="lines+markers",
        name="Eval Loss"
    ))
    fig.update_layout(
        title="Training vs. Evaluation Loss per Epoch",
        xaxis_title="Epoch",
        yaxis_title="Loss"
    )

    fig_path = os.path.join(output_dir, "loss_plot.html")
    fig.write_html(fig_path)

if __name__ == "__main__":
    # 1) Fetch raw documents (no custom cleaning)
    raw = return_documents("reddit", "noburp_all", ["noburp"])
    texts = [str(doc) for doc in raw if doc]

    # 2) Create a Dataset from raw texts
    ds = prepare_dataset(texts)
    print("Raw dataset size (before split):", len(ds))   

    # 3) Generate a timestamp string
    timestamp = datetime.now().strftime("%m_%d_%H_%M")

    # 4) Fine-tune each model in its own folder named finetuned_<short_name>_<timestamp>
    script_parent = os.path.dirname(os.path.abspath(__file__))
    for short_name, checkpoint in MODELS:
        outdir = os.path.join(script_parent, f"finetuned_{short_name}_{timestamp}")
        os.makedirs(outdir, exist_ok=True)
        print(f"Fine-tuning {checkpoint} → {outdir}")
        fine_tune_mlm(checkpoint, ds, outdir)