import re
TERM_CATEGORY_DICT = {
    "Experience with RCPD": ['Faux pas', 'Anxiety', 'Social-anxiety', 'Flare-ups', 'Misdiagnosis', 'R-CPD', 'Isolation'], 
    "Symptoms": ['Vomit', 'Vomit air', 'Acid reflux', 'Air sick', 'Air burp', 'Air vomiting', 'Cough', 'Croaks', 'Chest pain', 'Gag', 'Gassy', 'Gurgling', 'Gurgles', 'Gas', 'Croaking', 'Internal burps', 'Pressure', 'Bloating', 'Retching', 'Reflux', 'Regurgitation', 'Symptoms', 'Shortness of breath', 'Throwing up', 'Throat tightness', 'Throat gurgles', 'Hiccups', 'Supragastric belching', 'Indigestion', 'Difficulty breathing', 'Gastrointestinal distress'],
    "Self-treatment methods": ['Chamomile', 'Tea', 'Exercises', 'Gas-X', 'Famotidine', 'Fizzy drinks', 'Omeprazole', 'Neck turning', 'Self-taught', 'Self-curing', 'Shaker', 'Pelvic tilt', 'Mounjaro', 'antacids', 'kiss the ceiling', 'Self-cured', 'Rapid Drink Challenge'],
    "Doctor based interventions": ['(Pre,post) -Botox', 'In-office procedure', 'surgery', 'Anesthesia', 'procedure', 'Units', 'Xeomin', 'Esophageal dilation', 'Injections', 'Saline'],
    "Associated possible conditions": ['hiatal hernia', 'Dyspepsia', 'GERD', 'Emetophobia', 'Abdomino phrenic dyssynergia (APD)', 'GI disorder', 'irritable bowel syndrome'], 
    "Who to seek for diagnosis/treatment": ['ENT', 'Gastroenterologist', 'Laryngologist', 'Otolaryngologist', 'PCP', 'Specialist', 'Insurance'], 
    "Diagnostic Steps": ['Endoscopy', 'Nasoendoscopy', 'Swallow tests', 'esophageal examination', 'HRM', 'Manometry', 'Fluoroscopy', 'Imaging', 'Barium swallow', 'FEES test'],
    "General Anatomy and Physiology Involved": ['Throat muscles', 'GI', 'Cricoid', 'Swallow', 'Peristalsis', 'Retrograde Cricopharyngeal Dysfunction']
}

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

# words = extract_highlighted_words('user_posts.txt')
# print(words)
if __name__ == "__main__":
    pass