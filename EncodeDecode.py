import json
from TextPreprocessing import TextPreprocessing
import numpy as np

tp = TextPreprocessing()


class EncodDecode:
    
    def __init__(self):
        with open("json_data/vocab.json", "r", encoding="utf-8") as file:
            self.token_to_id = json.load(file)

        self.id_to_token = {
            str(value): key
            for key, value in self.token_to_id.items()
        }

        with open("json_data/pairs.json", "r", encoding="utf-8") as file:
            self.pairs = json.load(file)


    def encode(self, text):
        """Converts a text string into a list of numeric token IDs."""

        tokens = tp.clean_and_tokenize(text)

        unk_id = self.token_to_id.get("<UNK>", 0)

        return [
            self.token_to_id.get(tok, unk_id)
            for tok in tokens
        ]


    def encode_decoder_input(self, text):
        """Adds <START> token before encoded decoder text."""

        return (
            [self.token_to_id["<START>"]]
            + self.encode(text)
        )


    def encode_decoder_target(self, text):
        """Adds <END> token after encoded decoder text."""

        return (
            self.encode(text)
            + [self.token_to_id["<END>"]]
        )


    def encode_all_pairs(self):
        """Encodes all encoder-decoder pairs."""

        encoded_pairs = []

        for pair in self.pairs:

            encoder_ids = self.encode(
                pair["encoder"]
            )

            decoder_input_ids = self.encode_decoder_input(
                pair["decoder"]
            )

            decoder_target_ids = self.encode_decoder_target(
                pair["decoder"]
            )

            encoded_pairs.append({
                "encoder_ids": encoder_ids,
                "decoder_input_ids": decoder_input_ids,
                "decoder_target_ids": decoder_target_ids
            })

        return encoded_pairs


    def save_encoded_pairs(self):
        """Encodes all pairs and saves them into a JSON file."""

        encoded_pairs = self.encode_all_pairs()

        with open(
            "json_data/encoded_pairs.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                encoded_pairs,
                file,
                indent=2
            )

        print("Encoded pairs saved successfully!")

    def load_encoded_pair(self):
        with open("json_data/encoded_pairs.json","r") as f:
            enc = json.load(f)
        return enc
    def pad_sequences(self,sequences, pad_id=0):
        max_len = max(len(seq) for seq in sequences)
        padded = np.full((len(sequences), max_len), pad_id, dtype=np.int64)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        return padded

    def make_padding_mask(self,padded_ids, pad_id=0):
        return padded_ids != pad_id

    def make_batches(self,encoded_pairs,batch_size = 16,shuffle = True,seed = 42):
        indices = list(range(len(encoded_pairs)))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch = [encoded_pairs[i] for i in batch_idx]

            encoder_ids = self.pad_sequences([b["encoder_ids"] for b in batch])
            decoder_input_ids = self.pad_sequences([b["decoder_input_ids"] for b in batch])
            decoder_target_ids = self.pad_sequences([b["decoder_target_ids"] for b in batch])

            encoder_mask = self.make_padding_mask(encoder_ids)
            decoder_mask = self.make_padding_mask(decoder_input_ids)

            yield {
                "encoder_ids": encoder_ids,
                "decoder_input_ids": decoder_input_ids,
                "decoder_target_ids": decoder_target_ids,
                "encoder_mask": encoder_mask,
                "decoder_mask": decoder_mask,
            }


