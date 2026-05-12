from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
MAX_BATCH_SIZE = 100

client = OpenAI()


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        raise ValueError("texts list cannot be empty")

    if len(texts) > MAX_BATCH_SIZE:
        raise ValueError(f"batch size cannot exceed {MAX_BATCH_SIZE}")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    vectors = [item.embedding for item in response.data]

    return vectors


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
