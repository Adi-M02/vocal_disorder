import initial_analyses as ia
import sys
sys.path.append("/local/disk2/not_backed_up/amukundan/research/vocal_disorder")
import query_mongo as query

MAIN_RCPD_SUBREDDITS = ["noburp"]

import os
import re
import nltk
import html
from pymongo import MongoClient
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
import torch

################################
# 1) LOAD DATA FROM MONGODB
################################
def get_noburp_posts():
    """
    Fetches all NoBurp (RCPD-related) posts from a MongoDB collection.
    Returns a list of raw post strings.
    """
    try:
        posts = list(query.get_posts_by_subreddits(MAIN_RCPD_SUBREDDITS))

        # Extract and clean post texts
        raw_texts = []
        for post in posts:
            if "selftext" in post and post["selftext"].strip():
                text = post["selftext"]
                # Unescape HTML
                text = html.unescape(text)
                # Basic cleaning: remove odd whitespace
                text = re.sub(r"\s+", " ", text).strip()
                raw_texts.append(text)

        return raw_texts
    except:
        pass


################################
# 2) DATASET PREPARATION
################################
def create_dataset_from_posts(posts):
    """
    Takes a list of text posts and creates a Hugging Face Dataset object.
    """
    # The Dataset requires a dictionary of lists
    return Dataset.from_dict({"text": posts})

################################
# 3) PRETRAINING PREP
################################
def tokenize_function(examples, tokenizer, max_length=128):
    """
    Tokenize text for masked language modeling.
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"  # or 'longest' if you prefer dynamic padding
    )

################################
# 4) MAIN: CONTINUE PRETRAINING
################################
def main():
    # Download NLTK stopwords if you need them (optional)
    nltk.download("stopwords")

    # 1) Load data from MongoDB
    raw_posts = get_noburp_posts()  # adjust args if needed

    if not raw_posts:
        print("No NoBurp posts found in the database.")
        return

    print(f"Loaded {len(raw_posts)} NoBurp posts from MongoDB.")

    # 2) Create a Hugging Face Dataset
    dataset = create_dataset_from_posts(raw_posts)

    # 3) Load BioClinicalBERT and its tokenizer
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # 4) Tokenize the dataset
    def hf_tokenize(batch):
        return tokenize_function(batch, tokenizer)

    tokenized_dataset = dataset.map(hf_tokenize, batched=True, remove_columns=["text"])

    # 5) Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # 6) Split into train & validation sets (e.g., 90/10 split)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(eval_dataset)}")

    # 7) Set up training arguments
    training_args = TrainingArguments(
        output_dir="bioclinicalbert_noburp_pretrained",
        overwrite_output_dir=True,
        num_train_epochs=3,            # Adjust based on dataset size
        per_device_train_batch_size=8, # Adjust for GPU memory
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=500,
        report_to="none",             # or "tensorboard" if you want logging
        fp16=torch.cuda.is_available(),  # Use FP16 if you have a GPU with half-precision
    )

    # 8) Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    # 9) Train the model (continue pretraining)
    trainer.train()

    # 10) Save final model
    trainer.save_model("bioclinicalbert_noburp_pretrained/final_model")

    print("Pretraining completed! Model saved to bioclinicalbert_noburp_pretrained/final_model")

if __name__ == "__main__":
    main()

