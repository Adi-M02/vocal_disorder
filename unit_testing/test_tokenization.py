import sys
import pytest

# make sure we can import the tokenizer module
sys.path.append('../vocal_disorder')
from tokenizer import clean_and_tokenize

@pytest.mark.parametrize("input_text, expected_tokens", [
    # basic splitting and lowercasing
    ("This is a test.", ["this", "is", "a", "test"]),
    ("hello, world! test?", ["hello", "world", "test"]),
    # apostrophes stripped
    ("body's don't O'Reilly", ["bodys", "dont", "oreilly"]),
    # slash and hyphen handling
    ("slash/hyphen-test", ["slash", "hyphen", "test"]),
    # subreddit and user mention preserved
    ("Check r/test and u/User r/noburp r/testing", ["check", "r/test", "and", "u/user", "r/noburp", "r/testing"]),
    # URL removal
    ("Visit https://example.com/page for info", ["visit", "for", "info"]),
    ("Go to www.example.org now", ["go", "to", "now"]),
    # unicode normalization and dash/quote replacement
    ("“Quoted”—and…ellipsis", ["quoted", "and", "ellipsis"]),
    ("a–b—c", ["a", "b", "c"]),
    ("wait…now", ["wait", "now"]),
    ("a\u200Bb\u00A0c", ["a", "b", "c"]),
    # diacritics removed
    ("café naïve résumé", ["cafe", "naive", "resume"]),
        ("I paid $5.00 for 3.14 pies (100%)", 
     ["i","paid","5","00","for","3","14","pies","100"]),
    ("Time is 23:59 now", ["time","is","23","59","now"]),
    ("**bold** and _italic_ and `code()`", ["bold","and","italic","and","code"]),
    ("Contact me at user@example.com", ["contact","me","at","user","example","com"]),
    ("/u/spez commented in r/AmItheAsshole/subthread", 
     ["u/spez","commented","in","r/amitheasshole","subthread"]),
    ("r/test/subtest", ["r/test","subtest"]),
    ("u/User/test", ["u/user","test"]),
    ("test_case and snake_case", ["test","case","and","snake","case"]),
    ("😂😂 laugh 🙂 :-) :P", ["laugh"]),
    ("Look (here): [click]!", ["look","here","click"]),
    ("‘quote’—dash–dash", ["quote","dash","dash"]),
    ("中文测试 English", ["english"]),
    ("rock 'n' roll", ["rock","n","roll"]),
])
def test_clean_and_tokenize_various_cases(input_text, expected_tokens):
    tokens = clean_and_tokenize(input_text)
    assert tokens == expected_tokens