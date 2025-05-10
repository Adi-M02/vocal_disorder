
import sys
sys.path.append("/local/disk2/not_backed_up/amukundan/research/vocal_disorder")
import query_mongo as query


GROUND_TRUTHS = "vocabulary_evaluation/manual_terms.txt"
USERNAMES = ["freddiethecalathea", "Many_Pomegranate_566", "rpesce518", "kinglgw", "mjh59"]

posts = get_posts_by_users