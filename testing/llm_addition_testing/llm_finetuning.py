#!/usr/bin/env python3
# train_rcpd_qlora_unsloth_ddp_packing.py
"""
QLoRA DAPT for RCPD corpus on Llama-3.3-70B-Instruct (Unsloth, DDP + Sequence Packing)

- Launch with torchrun (1 process per GPU)
- Each process loads 4-bit model directly on its LOCAL_RANK GPU (device_map=f"cuda:{local_rank}")
- Sequence packing (packing=True) with max_seq_length=4096
- QLoRA adapters + gradient checkpointing
- Anti-forgetting: conservative LR, lora_dropout, weight decay, grad clip, early stopping, optional small replay
- Loss computed on logits.device and accepts Unsloth kwargs to avoid device mismatch
- Saves only LoRA adapters + tokenizer; base weights unchanged
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import List, Optional

# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ngram_phraser_dir", type=str, required=True,
                   help="Dir containing your bigram/trigram phrasers.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Base dir to save outputs/logs/LoRA.")
    p.add_argument("--num_epochs", type=int, default=1,
                   help="Few epochs reduce forgetting; 1–2 typical for DAPT.")
    p.add_argument("--learning_rate", type=float, default=1e-4,
                   help="Conservative LR to protect general knowledge.")
    p.add_argument("--grad_accum", type=int, default=4,
                   help="Gradient accumulation steps.")
    p.add_argument("--max_seq_length", type=int, default=4096,
                   help="Target chunk len; clamped to model max.")
    p.add_argument("--val_fraction", type=float, default=0.01,
                   help="Validation fraction (0..0.5).")
    p.add_argument("--eval_strategy", type=str, default="steps",
                   choices=["no", "steps", "epoch"], help="Evaluation strategy.")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--early_stopping_patience", type=int, default=5,
                   help="Stop if eval_loss doesn't improve for N evals.")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=0.3)
    p.add_argument("--cache_dir", type=str, default=None)

    # Optional replay to reduce forgetting
    p.add_argument("--replay_txt", type=str, default=None,
                   help="Path to a local .txt of general text to mix in (optional).")
    p.add_argument("--replay_fraction", type=float, default=0.05,
                   help="Portion of training docs drawn from replay file (0..0.3).")

    # Torchrun passes this; letting argparse consume it avoids errors
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()

args = parse_args()

# ----------------- project imports -----------------
sys.path.append("../vocal_disorder")
from utils.load_and_process_docs import process_all_noburp
from utils.text_pipeline import remove_unigram_stopwords
from testing.test_ngram_generation import load_phrasers_from_dir, apply_ngrams

def process_ngram_docs(ngram_phraser_dir: str) -> List[str]:
    docs = process_all_noburp(stoplist=False)
    bigram, trigram = load_phrasers_from_dir(ngram_phraser_dir)
    out = []
    for doc in docs:
        doc = apply_ngrams(doc, (bigram, trigram))
        doc = remove_unigram_stopwords(doc)
        out.append(doc)
    return out

# ----------------- libs (Unsloth BEFORE Transformers) -----------------
import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from unsloth import FastLanguageModel, is_bfloat16_supported  # <-- Unsloth first

from datasets import Dataset
from datasets import disable_caching as hf_disable_caching
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
    TrainerCallback,
)
from trl import SFTTrainer

# ----------------- utils -----------------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        ensure_dir(str(Path(csv_path).parent))
        if not Path(csv_path).exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "train_loss", "eval_loss"])
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([state.global_step, logs.get("loss"), logs.get("eval_loss")])

def read_replay_lines(path: str, max_lines: Optional[int] = None) -> List[str]:
    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
            if max_lines and len(lines) >= max_lines:
                break
    return lines

def normalize_to_text(docs: List) -> List[str]:
    """Coerce list/tuple tokens into strings; drop empties/very short."""
    cleaned = []
    for d in docs:
        if isinstance(d, (list, tuple)):
            d = " ".join(str(tok) for tok in d)
        elif isinstance(d, bytes):
            d = d.decode("utf-8", "ignore")
        elif not isinstance(d, str):
            d = str(d)
        d = d.strip()
        if d:
            cleaned.append(d)
    return [t for t in cleaned if len(t.split()) >= 3]

# ----------------- custom Trainer (safe loss, accepts Unsloth kwargs) -----------------
class MPFriendlySFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)    # avoid model's internal fast-loss path
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        else:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs.get("loss")
        return (loss, outputs) if return_outputs else loss

# ----------------- main -----------------
def main():
    set_seed(3407)
    hf_disable_caching()

    # Torchrun sets LOCAL_RANK; map each process to proper GPU
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank != -1 else 0))
    torch.cuda.set_device(local_rank)

    ensure_dir(args.output_dir)
    run_dir = os.path.join(args.output_dir, f"{args.num_epochs}_epochs")
    ensure_dir(run_dir)

    # 1) Load + normalize corpus
    if local_rank == 0: print("[data] loading & preprocessing RCPD corpus...")
    docs = process_ngram_docs(args.ngram_phraser_dir)
    docs = normalize_to_text(docs)

    # Optional replay to mitigate forgetting
    if args.replay_txt and Path(args.replay_txt).exists():
        replay_frac = max(0.0, min(0.3, float(args.replay_fraction)))
        n_replay = max(0, int(len(docs) * replay_frac))
        if n_replay > 0:
            rp_lines = read_replay_lines(args.replay_txt, max_lines=5 * n_replay)[:n_replay]
            if local_rank == 0:
                print(f"[replay] mixing {len(rp_lines)} generic lines from {args.replay_txt}")
            stride = max(1, len(docs) // max(1, len(rp_lines)))
            mixed, i_r = [], 0
            for i, d in enumerate(docs):
                mixed.append(d)
                if (i % stride) == 0 and i_r < len(rp_lines):
                    mixed.append(rp_lines[i_r]); i_r += 1
            docs = mixed

    # Small validation split
    val_frac = max(0.0, min(0.5, float(args.val_fraction)))
    n_total = len(docs)
    n_val = max(1, int(n_total * val_frac)) if n_total > 10 else 1
    val_texts = docs[:n_val]
    train_texts = docs[n_val:]
    if local_rank == 0:
        print(f"[split] train={len(train_texts)}  val={len(val_texts)}")

    # Raw text datasets (packing tokenizes internally)
    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds   = Dataset.from_dict({"text": val_texts})

    # 2) Resolve max context (config only)
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    cfg = AutoConfig.from_pretrained(model_id, cache_dir=args.cache_dir)
    max_ctx = getattr(cfg, "max_position_embeddings", None)
    max_seq = int(args.max_seq_length)
    if isinstance(max_ctx, int) and max_seq > max_ctx:
        if local_rank == 0:
            print(f"[warn] requested max_seq_length {max_seq} > model max {max_ctx}; clamping.")
        max_seq = max_ctx

    # 3) Tokenizer (used by packing)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=args.cache_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4) Load base model (4-bit) on THIS process's GPU
    if local_rank == 0: print(f"[model] loading 4-bit base on cuda:{local_rank} ...")
    model, _tok = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq,
        dtype=None,
        load_in_4bit=True,
        device_map=f"cuda:{local_rank}",   # load directly on local GPU
        cache_dir=args.cache_dir,
    )

    # 5) QLoRA adapters + gradient checkpointing
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha=64,
        lora_dropout=0.10,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # 6) TrainingArguments (DDP-friendly)
    use_bf16 = is_bfloat16_supported()
    training_args = TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=1,      # 70B@4k on A6000: keep small; raise if VRAM allows
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=args.logging_steps,

        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=(args.eval_strategy != "no"),
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,

        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        # DDP specifics
        ddp_find_unused_parameters=False,
        report_to="none",
        seed=3407,
    )

    # 7) SFTTrainer with **sequence packing**
    trainer = MPFriendlySFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,            # raw text; TRL packs internally
        eval_dataset=val_ds,
        data_collator=None,                # let TRL handle packing/collation
        max_seq_length=max_seq,
        packing=True,                      # <-- key
        dataset_text_field="text",
        args=training_args,
    )

    # Callbacks
    metrics_csv = os.path.join(run_dir, "metrics_log_rank{}.csv".format(local_rank))
    trainer.add_callback(MetricsLoggerCallback(metrics_csv))
    if args.eval_strategy != "no":
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    # Helpful stats (rank 0 only)
    if local_rank == 0:
        import math
        # With packing, steps/epoch is dynamic; this is a rough pretrain estimate
        eff_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        print(f"[info] DDP world_size={int(os.environ.get('WORLD_SIZE', '1'))}  local_rank={local_rank}  eff_batch={eff_batch}")
        print("[train] starting QLoRA fine-tuning with DDP + SEQUENCE PACKING ...")

    trainer.train(resume_from_checkpoint=True)

    if local_rank == 0:
        print("[train] done.")

    # 8) Save LoRA adapters + tokenizer (rank 0 only to avoid clobber)
    if local_rank == 0:
        lora_dir = os.path.join(run_dir, "lora_model")
        ensure_dir(lora_dir)
        print(f"[save] saving adapters + tokenizer to: {lora_dir}")
        model.save_pretrained(lora_dir)
        tokenizer.save_pretrained(lora_dir)

        print("\n[WHERE OUTPUT IS WRITTEN]")
        print(f"- Best-run dir: {run_dir}")
        print(f"- Metrics CSV (per-rank): {run_dir}/metrics_log_rank*.csv")
        print(f"- LoRA adapters: {lora_dir}/adapter_model.safetensors + adapter_config.json")
        print(f"- Tokenizer: {lora_dir}/tokenizer.json (and friends)")
        print(f"- Checkpoints: {run_dir}/checkpoint-*/ (best loaded at end)")

if __name__ == "__main__":
    main()
