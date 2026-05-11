from pathlib import Path
from src.chunking import chunk_document

def main():
    # Read a long document from the corpus
    file_path = Path("corpus/git-worktrees.md")
    text = file_path.read_text(encoding="utf-8")

    chunks = chunk_document(text, document_name=file_path.name)

    print(f"Document: {file_path.name}")
    print(f"Source: {len(text)} characters")
    print(f"Produced {len(chunks)} chunks\n")

    for chunk in chunks:
        print(f"Chunk {chunk['chunk_index']}: "
              f"{chunk['token_count']} tokens, "
              f"{chunk['char_count']} chars")

if __name__ == "__main__":
    main()