import argparse
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

def main(input_file, output_file):
    tokenizer = BertTokenizerFast.from_pretrained("model_output")
    model = BertForSequenceClassification.from_pretrained("model_output")
    model.eval()

    df = pd.read_csv(input_file)
    sentiments = []
    confidences = []

    for text in df["text"]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            confidence, label = torch.max(probs, dim=0)

        sentiments.append("positive" if label.item() == 1 else "negative")
        confidences.append(float(confidence))

    df["predicted_sentiment"] = sentiments
    df["confidence"] = confidences
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    main(args.input_file, args.output_file)
