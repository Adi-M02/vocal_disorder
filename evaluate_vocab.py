import re

def extract_highlighted_words(file_path):
    """
    Reads a text file and extracts all highlighted words or phrases wrapped in 
    [[hl]] and [[/hl]] markers.

    Parameters:
        file_path (str): Path to the text file.
        
    Returns:
        list: A list of highlighted words/phrases.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Use regex to extract content between [[hl]] and [[/hl]]
    highlighted_words = re.findall(r'\[\[hl\]\](.*?)\[\[\/hl\]\]', text)
    return highlighted_words

words = extract_highlighted_words('user_posts.txt')
print(words)