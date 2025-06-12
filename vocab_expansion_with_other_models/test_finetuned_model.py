from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import pipeline

finetuned_dir = "vocab_expansion_with_other_models/finetuned_modernBERT-base_06_11_13_48"
tokenizer = AutoTokenizer.from_pretrained(finetuned_dir)
model = AutoModelForMaskedLM.from_pretrained(finetuned_dir)

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

print(fill_mask("I am having slow [MASK] since the injections"))