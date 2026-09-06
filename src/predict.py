from src.dataset import load_dataset, split_dataset
from src.baseline import DATASET_PATH


def main():
    rows = load_dataset(DATASET_PATH)
    train, _, _ = split_dataset(rows)

    train_texts = [row["text"] for row in train]
    train_labels = [row["intent"] for row in train]

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

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

    print("Enter restaurant requests. Type 'exit' to stop.\n")

    while True:
        text = input("Customer: ").strip()

        if text.lower() == "exit":
            break

        if not text:
            continue

        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        confidence = max(probabilities)

        print(f"Intent: {prediction}")
        print(f"Confidence: {confidence:.4f}\n")


if __name__ == "__main__":
    main()

    