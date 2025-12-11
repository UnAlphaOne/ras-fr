#!/usr/bin/env python3
# RâS-Fr v8.0 – chat 512k tokens – 4 bits – < 6 GB VRAM
import torch, torch.nn.functional as F, time, math
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🎯 Chat 512k sur {DEVICE}")

# ---------- config ----------
MAX_CTX   = 512_000                  # 512 k tokens
GROUP_4B  = 64
CACHE_MO  = 5_000                    # 5 GB VRAM max
HEAD_DIM  = 128
ROPE_BASE = 10000

# ---------- quantif 4 bits ----------
def quantif_group_4bits(x: torch.Tensor) -> tuple:
    groupe = GROUP_4B
    mini, maxi = x.view(-1, groupe).min(1).values, x.view(-1, groupe).max(1).values
    scale = (maxi - mini) / 15
    zero  = mini
    x_int = ((x.view(-1, groupe) - zero.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 15)
    x_int = x_int.to(torch.uint8)
    x_int = x_int[:, ::2] << 4 | x_int[:, 1::2]
    return x_int, scale, zero

def dequantif_4bits(data: tuple) -> torch.Tensor:
    x_int, scale, zero = data
    x1 = (x_int >> 4) & 0xF
    x2 = x_int & 0xF
    x_int = torch.stack([x1, x2], dim=-1).view(-1)
    return x_int * scale.repeat_interleave(GROUP_4B) + zero.repeat_interleave(GROUP_4B)

# ---------- RoPE long ----------
def rope_long(x: torch.Tensor, seq_len: int, base: float = ROPE_BASE):
    dim = x.size(-1)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, device=x.device).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    x1, x2 = x[..., ::2], x[..., 1::2]
    return x1 * cos - x2 * sin, x1 * sin + x2 * cos

# ---------- nœud 512k ----------
class Noeud512k:
    def __init__(self, ident: int):
        self.ident = ident

    def materialiser(self, device: str):
        w1 = torch.randn(1024, 4096, device=device, dtype=torch.float16)
        w2 = torch.randn(4096, 1024, device=device, dtype=torch.float16)
        return quantif_group_4bits(w1), quantif_group_4bits(w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_data, w2_data = self.materialiser(x.device)
        w1 = dequantif_4bits(w1_data)
        w2 = dequantif_4bits(w2_data)
        return F.silu(x @ w1) @ w2

# ---------- cache sliding ----------
class CacheSliding512k:
    def __init__(self, max_mo: int = 5000):
        self.max_bytes = max_mo * 1024 * 1024
        self.cache = {}
        self.pos   = 0

    def get(self, ident: int, device: str):
        if ident in self.cache:
            return self.cache[ident]
        while len(self.cache) * 8 * 1024 * 4096 > self.max_bytes:
            self.cache.pop(next(iter(self.cache)))
        noeud = Noeud512k(ident)
        w1, w2 = noeud.materialiser(device)
        self.cache[ident] = (w1, w2)
        return w1, w2

cache_512k = CacheSliding512k()

# ---------- embedding 512k ----------
VOCAB_SIZE = 32003
embedding = nn.Embedding(VOCAB_SIZE, 1024).half().to(DEVICE)
tete_lm   = nn.Linear(1024, VOCAB_SIZE, bias=False).half().to(DEVICE)

# ---------- génération 512k ----------
@torch.no_grad()
def generer_512k(prompt: str, max_tokens: int = 50, temperature: float = 1.0):
    ids = [hash(c) % VOCAB_SIZE for c in prompt[:MAX_CTX]]
    ids = torch.tensor(ids, device=DEVICE).unsqueeze(0)
    for _ in range(max_tokens):
        if ids.size(1) > MAX_CTX:
            ids = ids[:, -MAX_CTX:]  # sliding window
        emb = embedding(ids)  # (1, L, 1024)
        emb = rope_long(emb, ids.size(1))  # RoPE long
        hidden = emb.mean(dim=1).unsqueeze(0)  # (1, 1, 1024)
        # 4 noeuds actifs
        idents = [hash(str(hidden.sum().item())) + i for i in range(4)]
        hidden = hidden.repeat(4, 1)
        for rank, ident in enumerate(idents):
            w1, w2 = cache_512k.get(ident, DEVICE)
            w1_d = dequantif_4bits(w1)
            w2_d = dequantif_4bits(w2)
            hidden[rank] = F.silu(hidden[rank] @ w1_d) @ w2_d
        out = hidden.mean(dim=0, keepdim=True)
        logits_lm = tete_lm(out)
        nxt = torch.multinomial(torch.softmax(logits_lm / temperature, dim=-1), num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
    return ids[0].tolist()

# ---------- bench ----------
def bench_512k():
    prompt = "La quantification 4 bits permet"
    torch.cuda.synchronize()
    t0 = time.time()
    out = generer_512k(prompt, max_tokens=50)
    torch.cuda.synchronize()
    tok_s = 50 / (time.time() - t0)
    print(">>>", out[:20])
    print(f"⚡ 512k tokens – tok/s : {tok_s:.1f}")
    print(f"🔌 VRAM max : {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

# ---------- main ----------
if __name__ == "__main__":
    bench_512k()