import re
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

def main():
    dataset = load_dataset("imdb")
    df = pd.concat([
        pd.DataFrame(dataset["train"]),
        pd.DataFrame(dataset["test"])
    ])

    df["text"] = df["text"].apply(clean_text)
    df = df[["text", "label"]]

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42
    )

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("Preprocessing complete")

if __name__ == "__main__":
    main()
