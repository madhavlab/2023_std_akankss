import os
import hnswlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time

def function1(search_db, dump_dir, file_type="flac", chkpt_path="/Users/akanksha/Downloads/vq-wav2vec.pt"):
    # Step 1: Generate manifest file
    os.system(f"python /Users/akanksha/new/fairseq/examples/wav2vec/wav2vec_manifest.py \
        {search_db} \
        --dest {dump_dir} \
        --ext {file_type}")

    # Step 2: Featurize audio files
    os.system(f"python /Users/akanksha/new/fairseq/examples/wav2vec/vq-wav2vec_featurize.py \
        --data-dir {dump_dir}  \
        --output-dir {dump_dir} \
        --checkpoint {chkpt_path} \
        --split train valid \
        --extension tsv")

if __name__ == "__main__" and False:
    search_db = "/Users/akanksha/Downloads/ref/reference"
    dump_dir = "/Users/akanksha/Downloads/ref/reference"
    function1(search_db, dump_dir)

def read_text_data(file_path):
    csv_file_path = os.path.join(file_path, "train_tokens.csv")
    print(csv_file_path)
    df = pd.read_csv(csv_file_path)
    return df['Data'].tolist(), df['Filename'].tolist()


def function2(query_audio_path, dump_dir, K):
    # Load and preprocess text data
    search_list, search_names = read_text_data(dump_dir)
    query_list, query_names = read_text_data(query_audio_path)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(search_list)

    start_time = time.time()
    # Calculate cosine similarity between query and search documents
    cosine_similarities = cosine_similarity(vectorizer.transform(query_list), tfidf_matrix)
    end_time = time.time()
    query_time = end_time - start_time

    # Get indices of top-K matches for each query
    top_k_indices = np.argsort(cosine_similarities, axis=1)[:, -K:][:, ::-1]

    # Get paths or indices to the top-K matched audio files
    top_k_matches = [[(search_names[idx], cosine_similarities[i, idx]) for idx in indices] for i, indices in enumerate(top_k_indices)]

    return top_k_matches, query_time

# Example Usage:
if __name__ == "__main__":
    query_audio_path = "/Users/akanksha/Downloads/test_queries/query"
    dump_directory = "/Users/akanksha/Desktop/pap_cai"
    query_dir="/Users/akanksha/Downloads/test_queries/query"
    function1(query_audio_path, query_dir )
    top_k_matches, query_time = function2(query_audio_path, dump_directory, K=10)
    print("Top-K Matches:")
    for i, matches in enumerate(top_k_matches):
        print(f"Query {i + 1}:")
        for match in matches:
            print(f"{match[0]}: {match[1]}")
        print()
    print("Query Time:", query_time)
