import subprocess
import itertools

# fixed arguments
TERMS       = "rcpd_terms_6_5.json"
MODEL_DIR   = "word2vec_expansion/word2vec_06_06_20_49"
MANUAL_DIR  = "vocabulary_evaluation/updated_manual_terms_6_3"
SPELLCHECK  = True  # set False to remove --spellcheck

# grid ranges
TOPK_VALUES  = list(range(25, 80, 5))      # 25,30,...,75
FREQ_VALUES  = [i/100 for i in range(15, 0, -1)]  # 0.15,0.14,...,0.01

SCRIPT_PATH = "word2vec_expansion/word2vec_expansion_lemmatized.py"

for topk, freq in itertools.product(TOPK_VALUES, FREQ_VALUES):
    cmd = [
        "python", SCRIPT_PATH,
        "--terms", TERMS,
        "--model_dir", MODEL_DIR,
        "--topk", str(topk),
        "--freq_threshold", str(freq),
        "--manual_dir", MANUAL_DIR,
        "--spellcheck", 
        
    ]

    print(f"\n→ Running with topk={topk}, freq_threshold={freq:.2f}")
    subprocess.run(cmd, check=True)