import os
from nltk.corpus import wordnet
import nltk

# If you haven’t already downloaded the WordNet data, uncomment these:
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# 1) Path to your file of comma-separated phrases
file_path = 'vocabulary_evaluation/updated_manual_terms_6_12/manual_terms.txt'

# 2) Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 3) Split on commas, strip whitespace, then split each phrase into words
phrases = [p.strip() for p in text.split(',') if p.strip()]
words = set()
for phrase in phrases:
    for token in phrase.split():
        words.add(token.lower())

# 4) Check membership in WordNet
in_wordnet     = {w for w in words if wordnet.synsets(w)}
not_in_wordnet = words - in_wordnet

# 5) Report
print(f"Total unique one-word terms: {len(words)}")
print(f"Terms in WordNet:         {len(in_wordnet)}")
print(f"Terms missing from WordNet: {len(not_in_wordnet)}")
print("\nExamples missing from WordNet:")
for w in list(not_in_wordnet):
    print(" ", w)
