from src.dataset import load_dataset, split_dataset


DATASET_PATH = "data/raw/restaurant_intents_combined.jsonl"


def main():
    rows = load_dataset(DATASET_PATH)
    train, validation, test = split_dataset(rows)

    print(f"Total examples: {len(rows)}")
    print(f"Training examples: {len(train)}")
    print(f"Validation examples: {len(validation)}")
    print(f"Test examples: {len(test)}")

    assert len(rows) == 600
    assert len(train) == 480
    assert len(validation) == 60
    assert len(test) == 60

    print("\nDataset loading and splitting succeeded.")


if __name__ == "__main__":
    main()
