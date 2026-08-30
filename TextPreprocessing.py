import json
import re

class TextPreprocessing:

    def __init__(self, file_path="json_data/pairs.json"):
        self.file_path = file_path
        self.pairs = []
        self.token_to_id = {}
        self.id_to_token = {}

    def clean_and_tokenize(self, text):
        text = text.lower()
        text = re.sub(r"([?.!,])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split(" ")

    def load_data(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.pairs = json.load(f)

    def build_vocab(self):
        special_tokens = ["<PAD>", "<START>", "<END>", "<UNK>"]
        self.token_to_id = {
            tok: i for i, tok in enumerate(special_tokens)
        }

        for pair in self.pairs:
            for text in [pair["encoder"], pair["decoder"]]:
                for token in self.clean_and_tokenize(text):
                    if token not in self.token_to_id:
                        self.token_to_id[token] = len(self.token_to_id)

        self.id_to_token = {
            i: tok for tok, i in self.token_to_id.items()
        }

    def save_vocab(self, output_path="json_data/id_to_token.json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.id_to_token, f, indent=2)

    def run(self):
        self.load_data()
        self.build_vocab()

        print("Vocab size:", len(self.token_to_id))

        self.save_vocab()


