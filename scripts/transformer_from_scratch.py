"""A hand-written Transformer (encoder-decoder) for understanding the source.

Implements the "Attention Is All You Need" architecture from scratch using only the
basic building blocks (`nn.Linear`, `nn.LayerNorm`, `nn.Embedding`, `nn.Dropout`) — no
`nn.Transformer`, `nn.MultiheadAttention`, or `F.scaled_dot_product_attention` — and
trains it on a tiny sequence-to-sequence task to prove the attention mechanism learns.

Why this matters here: LeRobot's ACT policy (``train_act.py``) is exactly this
architecture. ``modeling_act.py`` has a transformer *encoder* (reads image + joint-state
tokens) and a transformer *decoder* (cross-attends into the encoder, then regresses an
action chunk). Walking through this file is walking through the core of that source.

Usage:
    python scripts/transformer_from_scratch.py --task copy --steps 1500
    python scripts/transformer_from_scratch.py --task reverse --steps 3000
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(q, k, v, mask=None):
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V.

    q, k, v are (batch, heads, seq, d_head). `mask` is (batch, 1, S_q, S_k) with
    `0` where an entry must be excluded (set to -inf before softmax).
    """
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


class MultiHeadAttention(nn.Module):
    """Split the d_model into h heads, attend independently, then recombine."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x):  # (B, S, D) -> (B, H, S, d_head)
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

    def _merge(self, x):  # (B, H, S, d_head) -> (B, S, D)
        B, _, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, -1)

    def forward(self, query, key, value, mask=None):
        q = self._split(self.w_q(query))
        k = self._split(self.w_k(key))
        v = self._split(self.w_v(value))
        out = scaled_dot_product_attention(q, k, v, mask)
        out = self.dropout(self._merge(out))
        return self.w_o(out)


# ---------------------------------------------------------------------------
# Embedding / positional encoding
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal position signal, so attention (which is permutation-invariant)
    can see token order. Matches the scheme in the original paper."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x):  # x: (B, S, D)
        return x + self.pe[:, : x.size(1)]


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.net(x))


# ---------------------------------------------------------------------------
# Encoder / decoder stacks
# ---------------------------------------------------------------------------

# Note: pre-norm (LayerNorm before each sub-layer) trains more stably than the original
# post-norm; the residual ("x + ...") is what keeps gradients flowing through the stack.


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.ff(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 1. causal self-attention over the target so far
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 2. cross-attention: query = target, key/value = encoder memory
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.ff(x))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.enc_embed = nn.Embedding(src_vocab, d_model)
        self.dec_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos = PositionalEncoding(d_model)
        self.encoder = Encoder(d_model, n_heads, d_ff, n_enc_layers, dropout)
        self.decoder = Decoder(d_model, n_heads, d_ff, n_dec_layers, dropout)
        self.out = nn.Linear(d_model, tgt_vocab)

    def encode(self, src, src_mask=None):
        return self.encoder(self.pos(self.enc_embed(src)), src_mask)

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        return self.decoder(self.pos(self.dec_embed(tgt)), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        return self.out(self.decode(tgt, memory, src_mask, tgt_mask))


# ---------------------------------------------------------------------------
# Toy task + training
# ---------------------------------------------------------------------------

BOS, EOS = 10, 11  # vocab: digits 0-9 plus these two
VOCAB = 12


def causal_mask(seq_len: int, device) -> torch.Tensor:
    """(1, 1, L, L) lower-triangular mask so position i can only see positions <= i."""
    return (
        torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        .unsqueeze(0)
        .unsqueeze(0)
    )


def make_batch(batch_size: int, seq_len: int, task: str, device):
    digits = torch.randint(0, 10, (batch_size, seq_len), device=device)
    src = torch.cat([torch.full((batch_size, 1), BOS, device=device), digits], dim=1)  # (B, S+1)
    if task == "reverse":
        digits = torch.flip(digits, dims=[1])
    # teacher forcing: decoder input = [BOS] + digits ; labels = digits + [EOS]
    tgt_in = torch.cat([torch.full((batch_size, 1), BOS, device=device), digits], dim=1)
    tgt_out = torch.cat([digits, torch.full((batch_size, 1), EOS, device=device)], dim=1)
    return src, tgt_in, tgt_out


@torch.no_grad()
def greedy_decode(model, src, max_len, device):
    """Autoregressive decode: feed the encoder once, then emit one token at a time."""
    model.eval()
    memory = model.encode(src)
    tgt = torch.full((src.size(0), 1), BOS, device=device)
    for _ in range(max_len):
        logits = model.out(model.decode(tgt, memory, tgt_mask=causal_mask(tgt.size(1), device)))
        next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)  # (B, 1)
        tgt = torch.cat([tgt, next_tok], dim=1)
        if (next_tok == EOS).all():
            break
    return tgt


def accuracy(model, task, device, n_batches=50, seq_len=10):
    correct = total = 0
    for _ in range(n_batches):
        src, _, tgt_out = make_batch(32, seq_len, task, device)
        pred = greedy_decode(model, src, seq_len + 2, device)  # (B, T) incl. BOS
        pred_tokens = pred[:, 1:]  # drop the leading BOS
        # Compare the common prefix (early on the model may not emit EOS and can
        # over/under-shoot the target length, so only score the overlapping tokens).
        n = min(pred_tokens.size(1), tgt_out.size(1))
        correct += (pred_tokens[:, :n] == tgt_out[:, :n]).sum().item()
        total += 32 * n
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["copy", "reverse"], default="copy")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-freq", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model = Transformer(src_vocab=VOCAB, tgt_vocab=VOCAB).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Transformer ({n_params:,} params) on {device} | task={args.task}, seq_len={args.seq_len}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(1, args.steps + 1):
        model.train()
        src, tgt_in, tgt_out = make_batch(args.batch_size, args.seq_len, args.task, device)
        logits = model(src, tgt_in, tgt_mask=causal_mask(tgt_in.size(1), device))  # (B, T, vocab)
        loss = loss_fn(logits.reshape(-1, VOCAB), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_freq == 0 or step == 1:
            acc = accuracy(model, args.task, device, n_batches=20, seq_len=args.seq_len)
            print(f"  step {step:5d}/{args.steps} | loss {loss.item():.4f} | acc {100 * acc:.1f}%")

    print(f"Final accuracy ({args.task}): {100 * accuracy(model, args.task, device):.1f}%")


if __name__ == "__main__":
    main()
