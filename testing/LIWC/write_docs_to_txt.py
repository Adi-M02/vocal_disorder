import sys

sys.path.append('../vocal_disorder')

from query_mongo import return_documents
import csv

if __name__ == "__main__":
    # Example usage
    docs = return_documents(db_name='reddit', collection_name='noburp_all')
    output_path = '/local/disk2/not_backed_up/amukundan/research/LIWC_texts//noburp_all.txt'

    with open(output_path, 'w', encoding='utf-8') as txtfile:
        for doc in docs:
            txtfile.write(str(doc) + '\n')