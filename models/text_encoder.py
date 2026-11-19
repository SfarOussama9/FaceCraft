"""Transformer text encoder, trained from scratch (Vaswani et al., 2017).

Turns 77 token ids into:
  tokens [B, 77, dim]  one vector per token (the diffusion U-Net will attend to these with cross-attention)
  pooled [B, dim]      one vector for the whole sentence (mean of the real, non-padding tokens; used by the GAN)

Pieces:
  token embedding + learned position embedding  -> the model knows WHICH word and WHERE it is
  N pre-LayerNorm encoder layers (self-attention + MLP) -> every word looks at every other word
  padding mask -> padded positions are ignored by attention and by the pooling
"""
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 512, layers: int = 6, heads: int = 8, max_len: int = 77,
                 pad_id: int = 0, dropout: float = 0.1):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=4 * dim, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, token_ids: torch.Tensor):
        pad = token_ids == self.pad_id               # True where the position is padding
        pad[:, 0] = False                            # always keep the first token (avoids an all-masked row)
        x = self.token_emb(token_ids) + self.pos_emb[:, : token_ids.shape[1]]
        x = self.encoder(x, src_key_padding_mask=pad)
        tokens = self.norm(x)
        keep = (~pad).unsqueeze(-1).float()
        pooled = (tokens * keep).sum(1) / keep.sum(1)    # mean over real tokens only
        return tokens, pooled
