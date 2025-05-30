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
    TrainingArguments
)

sys.path.append('../vocal_disorder')
from query_mongo import return_documents
from tokenizer import clean_and_tokenize

# List of (short name, model checkpoint)
MODELS = [
    ("bert-base",      "bert-base-uncased"),
    ("bertweet",       "vinai/bertweet-base"),
    ("clinical-bert",  "emilyalsentzer/Bio_ClinicalBERT"),
    ("pubmed-bert",    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"),
]

def prepare_dataset(cleaned_docs: list[list[str]]) -> Dataset:
    # Re-join tokens into text strings
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
    # 0) Decide device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model     = AutoModelForMaskedLM.from_pretrained(model_name)

    # 1a) Ensure there's a real mask_token and pad_token
    special_tokens = {}
    # RoBERTa-style models (e.g. BERTweet) use "<mask>"
    if tokenizer.mask_token is None:
        special_tokens["mask_token"] = (
            "<mask>" if "roberta" in model_name or "bertweet" in model_name.lower()
            else "[MASK]"
        )
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = tokenizer.eos_token or "<pad>"
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))

    # 1b) Add all corpus tokens to tokenizer & resize again
    unique_tokens = set(tok for text in dataset["text"] for tok in text.split())
    new_tokens    = [tok for tok in unique_tokens if tok not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))

    # 1c) Move model to device *after* resizing embeddings
    model.to(device)

    # 2) Tokenize and split into train/eval
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True
        )

    splits   = dataset.train_test_split(test_size=0.1)
    train_ds = splits["train"].map(tokenize_fn, batched=True, remove_columns=["text"])
    eval_ds  = splits["test"].map(tokenize_fn,  batched=True, remove_columns=["text"])

    # 3) Data collator for dynamic masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )

    # 4) Training arguments
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

    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    # 6) Train + save
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 7) Save eval‐loss history
    log_history  = trainer.state.log_history
    eval_entries = [e for e in log_history if "eval_loss" in e]
    csv_path     = os.path.join(output_dir, "eval_loss.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "eval_loss"])
        for entry in eval_entries:
            writer.writerow([entry.get("epoch"), entry.get("eval_loss")])

if __name__ == "__main__":
    # 1) Fetch Reddit docs
    raw = return_documents("reddit", "noburp_all", ["noburp"])
    texts = []
    for doc in raw:
        if isinstance(doc, list):
            # Join tokens back into a raw text string
            text = " ".join(doc)
        else:
            text = str(doc)
        if text:
            texts.append(text)
    print(f"Total examples fetched: {len(texts)}")

    # 2) Clean & tokenize with custom tokenizer
    cleaned = [clean_and_tokenize(t) for t in texts]
    joined = [" ".join(tokens) for tokens in cleaned]

    # 3) Prepare Dataset
    ds = prepare_dataset(joined)

    # 4) Timestamped output directory
    timestamp = datetime.now().strftime("%m_%d_%H_%M")
    SELF_DIR   = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(SELF_DIR, f"finetuned_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)

    # 5) Fine-tune each model
    for short_name, checkpoint in MODELS:
        outdir = os.path.join(base_dir, short_name)
        os.makedirs(outdir, exist_ok=True)
        print(f"Fine-tuning {checkpoint} → {outdir}")
        fine_tune_mlm(
            model_name=checkpoint,
            dataset=ds,
            output_dir=outdir
        )
    # 6) Consolidate and visualize evaluation losses
    loss_data = {}
    for short_name, _ in MODELS:
        csv_path = os.path.join(base_dir, short_name, "eval_loss.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            loss_data[short_name] = df

    # 7) Build interactive line chart
    fig = go.Figure()
    for name, df in loss_data.items():
        fig.add_trace(go.Scatter(
            x=df["epoch"],
            y=df["eval_loss"],
            mode="lines+markers",
            name=name
        ))
    fig.update_layout(
        title="Evaluation Loss per Epoch for All Models",
        xaxis_title="Epoch",
        yaxis_title="Eval Loss"
    )

    # 8) Save HTML to base_dir
    html_path = os.path.join(base_dir, "all_eval_losses.html")
    fig.write_html(html_path)
    print(f"Saved consolidated loss plot to {html_path}")
