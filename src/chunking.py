import tiktoken

EMBEDDING_MODEL = "text-embedding-3-small"

tokenizer = tiktoken.encoding_for_model(EMBEDDING_MODEL)


def chunk_document(
    text: str,
    document_name: str,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 100,
):

    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap number must be less than chunk size")

    # tokenize
    token_ids = tokenizer.encode(text)
    total_tokens = len(token_ids)

    if total_tokens <= chunk_size_tokens:
        return [
            {
                "document_name": document_name,
                "chunk_index": 0,
                "text": text,
                "token_count": total_tokens,
                "char_count": len(text),
            }
        ]
    step = chunk_size_tokens - overlap_tokens
    chunks = []
    start = 0
    chunk_index = 0

    while start < total_tokens:
        end = start + chunk_size_tokens
        chunk_token_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_token_ids)

        chunks.append(
            {
                "document_name": document_name,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "token_count": len(chunk_token_ids),
                "char_count": len(chunk_text),
            }
        )

        if end >= total_tokens:
            break

        start += step
        chunk_index += 1

    return chunks
