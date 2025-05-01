import re
import spacy
import spacy.cli
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
from spellchecker import SpellChecker
import json
PROTECTED_TERMS = {"ibs", "ent", "ents", "gp", "op", "ptsd", "ocd", "rcpd"}
# Load the vocabulary JSON
with open("vocab_output/expanded_vocab_04_29_1906.json", "r") as f:
    vocab_data = json.load(f)
# Extract and flatten the list of vocabulary terms
custom_terms = set()
for category_terms in vocab_data["vocabulary"].values():
    for phrase in category_terms:
        # Lowercase and split compound terms (e.g., "Air_vomiting" → ["air", "vomiting"])
        words = phrase.lower().split('_')
        custom_terms.update(words)
spell = SpellChecker()
spell.word_frequency.load_words(custom_terms)

print ("Custom terms loaded:", custom_terms)

def lemmatize_word(word):
    if word in PROTECTED_TERMS:
        return word  # don't touch protected terms

    corrected = spell.correction(word) or word
    token = nlp(corrected)[0]
    lemma = token.lemma_

    return lemma

def lemmatize_term(term):
    term = term.lower()
    term = re.sub(r'[^a-z0-9\s\-]', '', term)

    tokens = re.split(r'\s+', term)

    normalized_parts = []
    for token in tokens:
        if '-' in token:
            subparts = token.split('-')
            norm_subparts = [lemmatize_word(sub) for sub in subparts]
            normalized = '-'.join(norm_subparts)
        else:
            normalized = lemmatize_word(token)
        normalized_parts.append(normalized)
    return ' '.join(normalized_parts)

words = ['able to burp', 'acid reflux', 'air vomiting', 'anesthesia', 'anxiety', 'bloating', 'botox', 'botox procedure', 'burping', 'chest pain', 'choking', 'cricoid massage', 'ent', 'excess wind', 'famotidine', 'gag', 'gurgle', 'gurgling', 'ibs', 'in-office procedure', 'inflammation', 'injection', 'laryngologist', 'lump', 'micro burp', 'microburps', 'no burp', 'pain', 'post-botox', 'pressure', 'procedure', 'regurgitation', 'sad', 'shaker exercise', 'slow swallow', 'slow swallowing', 'still burping', 'surgery', 'throwing up', 'tummy', 'uncomfortable', 'unit', 'vomit', 'vomiting', 'vomitting', 'air vomit', 'anaesthetic', 'appointment', 'bloated', 'botox technique', 'burp', 'burping uncontrollably', 'choke', 'cricopharyngeal myotomy', 'dosage', 'emetephobes', 'ents', 'forceful', 'gargling', 'gassy', 'gaviscon', 'gp', 'heave', 'laryngology', 'making myself sick', 'micro burping', 'miserable', 'mucus', 'myotomy', 'nasopharyngoscope', 'nausea', 'neck lift', 'neuropraxia', 'no burp syndrome', 'not burping', 'numb', 'numbness', 'op', 'painful', 'post-op', 'post-operative lingual neuropraxia', 'rcpd', 'recovery', 'regurgitated', 'relief', 'severity', 'side effect', 'sore', 'soreness', 'stomach pain', 'tenderness', 'tense', 'throat pain', 'throatox', 'throw up', 'tight', 'tight throat', 'vomited']
lemmatized_tuples = [(word, lemmatize_term(word)) for word in words]

# Print for inspection
for original, normalized in lemmatized_tuples:
    if original != normalized:
        print(f"{original:25} → {normalized}")