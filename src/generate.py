from src.embedding import client
from src.retrieval import retrieve

CHAT_MODEL = "gpt-4o-mini"


def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    context_blocks = []

    for chunk in chunks:
        context_blocks.append(
            f"[Source: {chunk['document_name']}, chunk {chunk['chunk_index']}]\n"
            f"{chunk['text']}"
        )

    context_text = "\n\n".join(context_blocks)

    system_message = (
        "You are a helpful assistant answering questions about a document corpus. "
        "Use only the context provided by the user. "
        "If the context does not contain enough information to answer the question, "
        "say so honestly and do not make up information."
    )

    user_message = f"Context:\n\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def generate_answer(query: str, k: int = 3) -> str:
    chunks = retrieve(query, k=k)
    messages = build_prompt(query, chunks)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.1,
    )

    return response.choices[0].message.content
