from src.dataset import load_dataset, split_dataset


DATASET_PATH = "data/raw/restaurant_intents.jsonl"


def main():
    rows = load_dataset(DATASET_PATH)
    train, validation, test = split_dataset(rows)

    print(f"Total examples: {len(rows)}")
    print(f"Training examples: {len(train)}")
    print(f"Validation examples: {len(validation)}")
    print(f"Test examples: {len(test)}")

    assert len(rows) == 200
    assert len(train) == 160
    assert len(validation) == 20
    assert len(test) == 20

    print("\nDataset loading and splitting succeeded.")


if __name__ == "__main__":
    main()