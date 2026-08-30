# first run csv_tojson.py then run this+6
from TextPreprocessing import TextPreprocessing

tp = TextPreprocessing("json_data/pairs.json")
tp.load_data()
tp.build_vocab()

import json
with open("json_data/vocab.json", "w", encoding="utf-8") as f:
    json.dump(tp.token_to_id, f, indent=2)

print(f"Vocab built: {len(tp.token_to_id)} tokens")