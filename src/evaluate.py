from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.dataset import load_dataset, split_dataset


DATASET_PATH = "data/raw/restaurant_intents.jsonl"


def main():
    rows = load_dataset(DATASET_PATH)
    train, _, test = split_dataset(rows)

    train_texts = [row["text"] for row in train]
    train_labels = [row["intent"] for row in train]

    test_texts = [row["text"] for row in test]
    test_labels = [row["intent"] for row in test]

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(train_texts, train_labels)
    predictions = model.predict(test_texts)

    print("\nIncorrect predictions:\n")

    incorrect_count = 0

    for text, actual, predicted in zip(
        test_texts,
        test_labels,
        predictions,
    ):
        if actual != predicted:
            incorrect_count += 1

            print(f"Text:      {text}")
            print(f"Actual:    {actual}")
            print(f"Predicted: {predicted}")
            print("-" * 60)

    print(f"\nIncorrect examples: {incorrect_count}/{len(test)}")


if __name__ == "__main__":
    main()