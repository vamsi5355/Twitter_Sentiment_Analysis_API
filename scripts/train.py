import json
import pandas as pd
from pathlib import Path
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = Path("model_output")
RESULTS_DIR = Path("results")

OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    accuracy = accuracy_score(labels, preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def main():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding=True,
            truncation=True
        )

    train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df).map(tokenize, batched=True)

    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )
    test_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="tmp",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        logging_steps=50,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(
            {
                "accuracy": metrics["eval_accuracy"],
                "precision": metrics["eval_precision"],
                "recall": metrics["eval_recall"],
                "f1_score": metrics["eval_f1_score"]
            },
            f,
            indent=2
        )

    with open(RESULTS_DIR / "run_summary.json", "w") as f:
        json.dump(
            {
                "hyperparameters": {
                    "model_name": MODEL_NAME,
                    "learning_rate": 2e-5,
                    "batch_size": 8,
                    "num_epochs": 2
                },
                "final_metrics": {
                    "accuracy": metrics["eval_accuracy"],
                    "f1_score": metrics["eval_f1_score"]
                }
            },
            f,
            indent=2
        )


if __name__ == "__main__":
    main()