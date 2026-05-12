import numpy as np

from src.embedding import embed_text
from src.storage import load_all_chunks


def retrieve(query: str, k: int = 3) -> list[dict]:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    query_vector = np.array(embed_text(query), dtype=np.float32)
    chunks = load_all_chunks()

    if not chunks:
        return []

    chunk_matrix = np.stack([chunk["embedding"] for chunk in chunks])

    dot_products = chunk_matrix @ query_vector
    chunk_norms = np.linalg.norm(chunk_matrix, axis=1)
    query_norm = np.linalg.norm(query_vector)

    similarities = dot_products / (chunk_norms * query_norm)

    top_k_indices = np.argsort(similarities)[::-1][:k]

    results = []

    for i in top_k_indices:
        result = dict(chunks[i])
        result["similarity"] = float(similarities[i])
        results.append(result)

    return results
