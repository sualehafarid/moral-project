import json
import argparse
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def main():
    parser = argparse.ArgumentParser(
        description="Calculate evaluation metrics from parsed evaluation results."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the parsed evaluation results JSON file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output metrics JSON file"
    )
    args = parser.parse_args()
    
    # Load the parsed evaluation records.
    with open(args.input_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    
    predictions = []
    references = []
    for rec in records:
        if rec.get("valid", False):
            predictions.append(rec.get("parsed_answer", -3))
            references.append(rec.get("reference", 0))
    
    if not predictions:
        print("No valid records found. Exiting.")
        return

    # Calculate evaluation metrics.
    accuracy = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average="weighted")
    cm = confusion_matrix(references, predictions, labels=[0, 1, 2]).tolist()
    
    # Build the result dictionary.
    metrics = {
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix": cm
    }
    
    # Write the result to the output file.
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print("Metrics written to", args.output_file)

if __name__ == "__main__":
    main()
