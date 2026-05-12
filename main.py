from src.generate import generate_answer


def main() -> None:
    query = "What's the current weather forecast in Tokyo?"
    answer = generate_answer(query, k=3)
    print(answer)


if __name__ == "__main__":
    main()