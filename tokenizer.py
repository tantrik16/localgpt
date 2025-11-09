from dataclasses import dataclass, asdict
import json
import re
from typing import List, Union
from pathlib import Path


@dataclass
class TokenizerConfig:
    vocab_size: int
    min_frequency: int
    special_tokens: List[str]
    max_length: int

    def to_dict(self) -> dict:
        """Convert the config to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TokenizerConfig":
        """Create a TokenizerConfig from a dictionary."""
        return cls(
            vocab_size=data["vocab_size"],
            min_frequency=data["min_frequency"],
            special_tokens=data["special_tokens"],
            max_length=data["max_length"],
        )

    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save the config to a JSON file."""
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> "TokenizerConfig":
        """Load a TokenizerConfig from a JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class BPETokenizer:
    def __init__(self, tokenizer_config: TokenizerConfig):
        self.tokenizer_config = tokenizer_config
        self.vocab = []
        self.merges = []
        self.special_tokens = tokenizer_config.special_tokens
        self.vocab_size = tokenizer_config.vocab_size
        self.min_frequency = tokenizer_config.min_frequency
        self.max_length = tokenizer_config.max_length

    def get_all_pair_freq(self, freq_dict: dict) -> dict:
        """Get all pair frequencies from the frequency dictionary."""
        pair_freq = {}
        for word, freq in freq_dict.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freq[pair] = pair_freq.get(pair, 0) + freq
        return pair_freq

    def get_next_merge_pair(self, freq_dict: dict) -> tuple:
        """Get the next merge pair from the frequency dictionary."""
        max_freq = 0
        max_pair = None
        pair_freq = self.get_all_pair_freq(freq_dict)
        for pair, freq in pair_freq.items():
            if freq > max_freq:
                max_freq = freq
                max_pair = pair
        return max_pair, max_freq

    def update_freq_dict(self, freq_dict: dict, pair: tuple, new_token: str) -> dict:
        """Update the frequency dictionary with the new token."""
        new_freq_dict = {}
        for word in freq_dict.keys():
            new_word = []
            i = 0
            while i < len(word) - 1:
                if word[i] + word[i + 1] == new_token:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            if i == len(word) - 1:
                new_word.append(word[i])
            new_freq_dict[tuple(new_word)] = freq_dict[word]
        return new_freq_dict

    def pre_tokenize_str(self, text: str):
        """
        Simplified version of GPT-2 ByteLevel pre_tokenizer.pre_tokenize_str.
        Returns list of (token, (start, end)) pairs.
        """
        return re.findall(r'\w+|[^\w\s]|[\s]+', text)

    def train(self, texts: List[str]) -> None:
        """Train the tokenizer on the given texts."""
        freq_dict = {}
        for text in texts:
            words = self.pre_tokenize_str(text)
            for word in words:
                freq_dict[tuple(word)] = freq_dict.get(tuple(word), 0) + 1

        token_freq = {}
        for word, freq in freq_dict.items():
            for token in word:
                token_freq[token] = token_freq.get(token, 0) + freq
        self.vocab = sorted(
            token_freq.keys(), key=lambda x: token_freq[x], reverse=True
        )

        while len(self.vocab) < self.vocab_size:
            pair, freq = self.get_next_merge_pair(freq_dict)
            # merge pair into a new token
            if pair is None or freq < self.min_frequency:
                break
            new_token = "".join(pair)
            if new_token in self.vocab:
                break
            self.vocab.append(new_token)
            freq_dict = self.update_freq_dict(freq_dict, pair, new_token)
            # add new token to merges
            self.merges.append((pair, freq))

    def tokenize(self, text: str) -> List[str]:
        """Tokenize the given text."""
        pass

    def detokenize(self, tokens: List[str]) -> str:
        """Detokenize the given tokens."""
        pass


if __name__ == "__main__":
    print("Training tokenizer...")
    tokenizer = BPETokenizer(
        TokenizerConfig(
            vocab_size=50,
            min_frequency=2,
            special_tokens=["<pad>", "<unk>"],
            max_length=100,
        )
    )
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    tokenizer.train(corpus)
    print ("Vocab Size:", len(tokenizer.vocab))
    print("Vocab:", tokenizer.vocab)
    print("Merges:", tokenizer.merges)
