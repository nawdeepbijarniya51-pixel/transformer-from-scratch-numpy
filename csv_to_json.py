import csv
import json

def csv_to_pairs_json(csv_filepath, json_filepath, delimiter=","):
    pairs = []
    with open(csv_filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) < 3:
                continue
            _, input_line, output_line = row[0], row[1], row[2]
            pairs.append({
                "encoder": input_line.strip(),
                "decoder": output_line.strip()
            })

    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(pairs)} pairs to {json_filepath}")

csv_to_pairs_json("csv_data/Conversation.csv", "json_data/pairs.json")