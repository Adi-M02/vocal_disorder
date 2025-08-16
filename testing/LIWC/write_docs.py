import sys
import csv

sys.path.append('../vocal_disorder')

from query_mongo import return_documents

if __name__ == "__main__":
    docs = return_documents(db_name='reddit', collection_name='noburp_all')
    output_path = '/local/disk2/not_backed_up/amukundan/research/LIWC_texts/noburp_all.csv'

    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['doc_id', 'text'])
        for idx, doc in enumerate(docs):
            writer.writerow([idx, str(doc)])