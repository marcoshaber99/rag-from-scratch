import sqlite3
from pathlib import Path

import numpy as np

from src.embedding import EMBEDDING_DIM

DB_PATH = Path("data/vectors.db")


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_name TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                token_count INTEGER NOT NULL,
                char_count INTEGER NOT NULL
            )
            """
        )


def save_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    with sqlite3.connect(DB_PATH) as conn:
        for chunk, embedding in zip(chunks, embeddings):
            embedding_array = np.array(embedding, dtype=np.float32)

            if embedding_array.shape != (EMBEDDING_DIM,):
                raise ValueError(
                    f"embedding must have dimension {EMBEDDING_DIM}, "
                    f"got {embedding_array.shape}"
                )

            embedding_blob = embedding_array.tobytes()

            conn.execute(
                """
                INSERT INTO chunks (
                    document_name,
                    chunk_index,
                    text,
                    embedding,
                    token_count,
                    char_count
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk["document_name"],
                    chunk["chunk_index"],
                    chunk["text"],
                    embedding_blob,
                    chunk["token_count"],
                    chunk["char_count"],
                ),
            )


def load_all_chunks() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT
                document_name,
                chunk_index,
                text,
                embedding,
                token_count,
                char_count
            FROM chunks
            """
        ).fetchall()

    loaded_chunks = []

    for row in rows:
        document_name, chunk_index, text, embedding_blob, token_count, char_count = row

        embedding_array = np.frombuffer(embedding_blob, dtype=np.float32)

        if embedding_array.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"loaded embedding must have dimension {EMBEDDING_DIM}, "
                f"got {embedding_array.shape}"
            )

        loaded_chunks.append(
            {
                "document_name": document_name,
                "chunk_index": chunk_index,
                "text": text,
                "embedding": embedding_array,
                "token_count": token_count,
                "char_count": char_count,
            }
        )

    return loaded_chunks


def clear_chunks() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM chunks")
