import os
import sys
from datetime import datetime
import json
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
from torch.utils.data import DataLoader

sys.path.append('../vocal_disorder')
from query_mongo import return_documents

# List of (short name, model checkpoint)
MODELS = [
    ("moderBERT-base", "answerdotai/ModernBERT-base")
    # ("bert-base",      "bert-base-uncased"),
    # ("bertweet",      "vinai/bertweet-base"),
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
    epochs: int = 4,
    batch_size: int = 2,
    max_length: int = 8192,
    mlm_probability: float = 0.15
):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # 1) Load tokenizer & model (with normalization for Bertweet)
    if "bertweet" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            normalization=True,
            use_fast=False
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # 2) Move model to device
    model.to(device)

    # 3) Tokenize and split into train/eval
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True
        )

    splits = dataset.train_test_split(test_size=0.1, seed=42)
    # Shuffle both train and test splits with seed 42
    splits["train"] = splits["train"].shuffle(seed=42)
    splits["test"] = splits["test"].shuffle(seed=42)
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

    # 4) Sanity check: no token ID >= vocab_size
    for split_name in ("train", "eval"):
        all_ids = tokenized[split_name]["input_ids"]
        max_id = max(max(seq) for seq in all_ids)
        if max_id >= model.config.vocab_size:
            raise ValueError(
                f"Token index {max_id} ≥ vocab_size {model.config.vocab_size}. "
                "Check tokenizer–model mismatch."
            )

    # 5) separate training and eval  collators
    train_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )
    eval_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 6) Training arguments
    has_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=has_cuda,
		ddp_find_unused_parameters=None,
        dataloader_num_workers=4,
        learning_rate=3e-5,
        lr_scheduler_type="linear",
        warmup_steps=1000,
        load_best_model_at_end=True,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=2500,
        save_total_limit=2,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
        seed=42
    )

    # ─── Write hyperparameters and training args to config.json ─────────────────
    config = {
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "mlm_probability": mlm_probability,
        "training_arguments": args.to_dict()
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    # 7) Trainer and eval collator helper
    class MLMEvalTrainer(Trainer):
        def get_eval_dataloader(self, eval_dataset):
            eval_sampler = self._get_eval_sampler(eval_dataset)
            return DataLoader(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                sampler=eval_sampler,
                collate_fn=eval_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
    trainer = MLMEvalTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        data_collator=train_collator,        # train→masking
    )

    # 8) Train, save model & tokenizer
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 9) Extract log history
    log_history = trainer.state.log_history

    # Separate training-loss and eval-loss entries
    train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_entries  = [e for e in log_history if "eval_loss" in e]

    # 10) Aggregate losses by epoch for plotting
    train_by_epoch = {}
    for entry in train_entries:
        ep = entry.get("epoch")
        loss = entry.get("loss")
        train_by_epoch[ep] = loss

    eval_by_epoch = {entry.get("epoch"): entry.get("eval_loss") for entry in eval_entries}

    train_epochs = sorted(train_by_epoch.keys())
    eval_epochs  = sorted(eval_by_epoch.keys())

    train_losses = [train_by_epoch[ep] for ep in train_epochs]
    eval_losses  = [eval_by_epoch[ep] for ep in eval_epochs]

    # 11) Build Plotly chart for both
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

    # 12) Save Plotly chart as HTML
    fig_path = os.path.join(output_dir, "loss_plot.html")
    fig.write_html(fig_path)


if __name__ == "__main__":
    # 1) Fetch raw documents (no custom cleaning)
    raw = return_documents("reddit", "noburp_all", ["noburp"])
    texts = [str(doc) for doc in raw if doc]

    # 2) Create a Dataset from raw texts
    ds = prepare_dataset(texts)

    # 3) Generate a timestamp string
    timestamp = datetime.now().strftime("%m_%d_%H_%M")

    # 4) Fine-tune each model in its own folder named finetuned_<short_name>_<timestamp>
    script_parent = os.path.dirname(os.path.abspath(__file__))
    for short_name, checkpoint in MODELS:
        outdir = os.path.join(script_parent, f"finetuned_{short_name}_{timestamp}")
        os.makedirs(outdir, exist_ok=True)
        print(f"Fine-tuning {checkpoint} → {outdir}")
        fine_tune_mlm(checkpoint, ds, outdir)
