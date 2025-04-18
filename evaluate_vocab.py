import re
import query_mongo as query
from collections import defaultdict
from expand_vocab_modular import preprocess_text

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

UPDATED_TERMS_WO_CATEGORIES = terms = [
    "burp_stomach_issues", "Flare-ups", "anxiety_induced", "Isolation", "bastions", "burps_case", "bulimic", "burps_stay", "anxiety_issue", "R-CPD", "burbs", "Misdiagnosis", "anxiety_gone", "anxiety_due", "Social-anxiety", "burp_anxiety_issues", "Anxiety", "anxiety_cause", "tummy", "gaviscon", "social_anxiety_also", "scouring",
    "vomit_burps", "Symptoms", "air_vomiting_etc", "Throwing up", "Supragastric belching", "Gastrointestinal distress", "vomiting_work", "Internal burps", "Gurgles", "fear_vomiting_r", "Shortness of breath", "vomiting_hold", "vomiting_bad", "Gurgling", "Pressure", "Air vomiting", "burps_vomit", "Vomit", "get_air_vomiting", "Hiccups", "Chest pain", "Indigestion", "vomiting_sound", "burps_symptoms", "cramps_gas", "Vomit air", "air_vomiting_throat", "Croaking", "burp_stomach_issues", "Gag", "burps_stomach", "Croaks", "Retching", "Cough", "Air sick", "Throat tightness", "vomiting_pressure", "Gassy", "hiccups_stomach", "Reflux", "sore_throat_etc", "Acid reflux", "vomiting_sounds", "Air burp", "Gas", "vomiting_try", "Throat gurgles", "air_vomiting_sound", "Difficulty breathing", "Bloating", "Regurgitation",
    "Omeprazole", "Shaker", "granada", "hicklin_treatment", "Neck turning", "Gas-X", "burp_lots", "Self-taught", "croaking_lot", "burps_day", "botox_drink", "croaks_lot", "botox_swallowing", "Chamomile", "today_burps", "veggies", "today_burp", "Rapid Drink Challenge", "Famotidine", "Exercises", "Pelvic tilt", "Fizzy drinks", "burp_everyday", "chug_water", "burps_afternoon", "Self-cured", "botox_wear", "burps_night", "Tea", "Self-curing", "Mounjaro", "antacids", "kiss the ceiling", "burp_mini", "tummy", "throat_rest_day",
    "Esophageal dilation", "procedure_always", "procedure_happy", "procedure", "In-office procedure", "procedure_thing", "procedure_hope", "procedure_something", "procedure_talking", "(Pre,post) -Botox", "procedure_sore", "procedure_waited", "back_work_procedure", "procedure_worried", "procedure_doctor", "nervous_procedure", "Injections", "Xeomin", "procedure_often", "procedure_get", "procedure_trying", "Anesthesia", "procedure_first", "procedure_give", "procedure_said", "surgery", "Units", "procedure_us", "procedure_help", "Saline",
    "ibs_coeliac_disease", "hi_everybody_dysmenorrhea", "reflux_gerd_symptoms", "Dyspepsia", "gastric_reflux", "emetophobia", "hiatal_hernia_gerd", "Emetophobia", "irritable bowel syndrome", "acid_reflux_esophogitis", "triathlon", "botox_dysphagia", "Abdomino phrenic dyssynergia (APD)", "acid_reflux_gerd", "celiac_disease_etc", "GERD", "dyssynergia", "anxiety_emetophobia", "dyspepsia", "hiatal hernia", "emetophobia_anxiety", "sibo_gastritis", "gastritis_gerd", "acid_reflux_gastritis", "anxiety_acid_reflux", "GI disorder", "phrenic_dyssynergia_apd",
    "bastian_say", "Specialist", "bastian_talk", "laryngologist", "bastian_seeing", "ent_two", "Insurance", "ent_looked", "ent_doctor", "england", "PCP", "bastien_office", "ent_consultant", "Laryngologist", "bastian_office", "Otolaryngologist", "bastian_everything", "ent_never", "Gastroenterologist", "bastians_research", "ent_specialist", "hicklin_consultant", "ent_actually", "bastian_doctors", "gaviscon", "gerd_think", "ENT",
    "endoscopy_get", "barium_swallow", "Barium swallow", "esophageal examination", "endoscopy_anyone", "endoscopy", "endoscopy_scheduled", "barium_swallow_endoscopy", "barium_test", "endoscopy_nothing", "Endoscopy", "today_barium_swallow", "swallow_gastroscopy", "endoscopy_make", "endoscopy_barium", "endoscopy_part", "endoscopy_check", "endoscopy_due", "swallow_test_endoscopy", "endoscopy_barium_swallow", "endoscopy_next", "FEES test", "endoscopy_look", "Nasoendoscopy", "Swallow tests", "HRM", "Imaging", "test_endoscopy", "Manometry", "Fluoroscopy",
    "gas_reflux", "cricoid_massage_help", "burp_stomach_issues", "cricoid_massage_hurt", "swallow_reflux", "reflux_get", "Cricoid", "Throat muscles", "Retrograde Cricopharyngeal Dysfunction", "slow_swallow_globus", "cricoid_massage_thing", "stomach_gi", "reflux_go", "cricoid_exercise", "cricoid_massage_video", "Swallow", "gas_tummy", "GI", "alas", "swallow_motility", "throat_cricoid_massage", "barium_swallow_motility", "vocal_cords_etc", "cricoids", "Peristalsis", "cringe"
]

users = ['ThinkSuccotash', 'ScratchGolfer1976', 'Mobile-Breakfast-526', 'Wrob88', 'Tornteddie', 'AmazingAd5243']
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

    manual_terms = extract_highlighted_words('user_posts.txt')
    posts = query.get_posts_by_users(users, ["noburp"])
    print(len(posts))
    dict_terms = [word.lower().replace('_', ' ') for word in UPDATED_TERMS_WO_CATEGORIES]
    dict_terms = set(dict_terms)
    manual_terms_set = set(manual_terms)

    # Initialize results structure
    term_occurrences = defaultdict(lambda: {'dictionary': 0, 'manual': 0, 'both': 0})

    # Check occurrences in posts
    for post in posts:
        post = post.get('selftext', '') + ' ' + post.get('title', '')
        post = preprocess_text(post)
        
        # Track matches
        dict_matched = set()
        manual_matched = set()
        
        for term in dict_terms:
            if re.search(r'\b{}\b'.format(re.escape(term)), post):
                dict_matched.add(term)
                
        for term in manual_terms_set:
            if re.search(r'\b{}\b'.format(re.escape(term)), post):
                manual_matched.add(term)
                
        # Record occurrences explicitly
        for term in dict_matched & manual_matched:
            term_occurrences[term]['both'] += 1
        for term in dict_matched - manual_matched:
            term_occurrences[term]['dictionary'] += 1
        for term in manual_matched - dict_matched:
            term_occurrences[term]['manual'] += 1

    agreement_terms = [term for term, counts in term_occurrences.items() if counts['both'] > 0]
    dict_only_terms = [term for term, counts in term_occurrences.items() if counts['dictionary'] > 0 and counts['both'] == 0]
    manual_only_terms = [term for term, counts in term_occurrences.items() if counts['manual'] > 0 and counts['both'] == 0]

    print("✅ Terms found by BOTH dictionary and manual list:", agreement_terms)
    print("📙 Terms found ONLY by dictionary:", dict_only_terms)
    print("✏️ Terms found ONLY by manual list:", manual_only_terms)
    print()
    print(manual_terms)