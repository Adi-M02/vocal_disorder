from gensim.models import Word2Vec
import numpy as np
from numpy import dot
from numpy.linalg import norm

def save_word2vec_embeddings(model: Word2Vec, output_path: str):
    with open(output_path, 'w') as f:
        for word in model.wv.index_to_key:
            vector = model.wv[word]
            vec_str = ' '.join(map(str, vector))
            f.write(f"{word} {vec_str}\n")

def load_embeddings(filepath: str) -> dict[str, np.ndarray]:
    embeddings = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(list(map(float, parts[1:])))
            embeddings[word] = vec
    return embeddings

def cosine_similarity(vec1, vec2) -> float:
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def euclidean_distance(vec1, vec2) -> float:
    return norm(vec1 - vec2)

if __name__ == "__main__":
    embeddings = load_embeddings("word2vec_expansion/word2vec_06_23_13_58/word2vec_cbow.txt")
    term1 = "noburp"
    term2 = "emetephobia"
    print(cosine_similarity(embeddings[term1], embeddings[term2]))
    print(euclidean_distance(embeddings[term1], embeddings[term2]))