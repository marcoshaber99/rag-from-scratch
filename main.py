from src.embedding import embed_text

def main():
    text = "Cyprus property regulations include VAT reductions for primary residences."
    print(f"Embedding text: {text!r}")
    print()

    vector = embed_text(text)

    print(f"Vector length: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
    print(f"Last 5 values: {vector[-5:]}")
    print(f"Sum: {sum(vector):.4f}")

if __name__ == "__main__":
    main()