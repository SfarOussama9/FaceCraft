"""Text -> token ids.

We reuse CLIP's BPE tokenizer: it is only a vocabulary lookup table (no neural network weights),
so using it does not break the "from scratch" rule. Our Transformer text encoder, trained from scratch,
learns what the tokens mean. Every caption becomes exactly 77 ids (padded / truncated), like CLIP.

HashTokenizer is a tiny offline fallback (used in tests and when there is no internet).
"""
import re

import torch

MAX_LEN = 77


class Tokenizer:
    """CLIP BPE tokenizer wrapper: list of strings -> LongTensor [B, 77]."""

    def __init__(self, name: str = "openai/clip-vit-base-patch32", max_len: int = MAX_LEN):
        from transformers import CLIPTokenizer
        self.tok = CLIPTokenizer.from_pretrained(name)
        self.max_len = max_len
        self.pad_id = self.tok.pad_token_id
        self.vocab_size = self.tok.vocab_size

    def __call__(self, texts: list) -> torch.Tensor:
        out = self.tok(list(texts), padding="max_length", truncation=True, max_length=self.max_len,
                       return_tensors="pt")
        return out["input_ids"].long()


class HashTokenizer:
    """Offline stand-in: lowercased words hashed into a fixed vocabulary (deterministic across runs)."""

    def __init__(self, vocab_size: int = 8192, max_len: int = MAX_LEN):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_id = 0

    def _id(self, word: str) -> int:
        h = 0
        for ch in word:                       # simple stable hash (Python's hash() changes per run)
            h = (h * 31 + ord(ch)) % 2_147_483_647
        return 1 + h % (self.vocab_size - 1)

    def __call__(self, texts: list) -> torch.Tensor:
        out = torch.full((len(texts), self.max_len), self.pad_id, dtype=torch.long)
        for i, t in enumerate(texts):
            ids = [self._id(w) for w in re.findall(r"[a-z']+", t.lower())][: self.max_len]
            if ids:
                out[i, : len(ids)] = torch.tensor(ids)
        return out


def make_tokenizer(kind: str = "clip"):
    """Factory used by training scripts: kind = "clip" (default) or "hash" (offline)."""
    return Tokenizer() if kind == "clip" else HashTokenizer()
