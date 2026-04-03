# residual_transformer_core.py
  
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ===================================================================
# RESIDUAL LEARNING BLOCKS
# Core of the residual learning hypothesis  
# Two preserved variants: vision (CNN) and general optimization (MLP).
# ===================================================================

class ResidualBlock(nn.Module):
    """CNN-style residual block (ResNet-style) with identity/projection shortcut.
    
    Reformulates mapping as H(x) = F(x) + x for easier optimization of deep conv nets.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out


class ResidualOptimizationBlock(nn.Module):
    """General/MLP-style residual block demonstrating the residual learning hypothesis.
    
    Learns F(x) so that H(x) = F(x) + x. If F learns zero, the block becomes identity.
    """
    def __init__(self, dimensions: int, hidden_dim: Optional[int] = None):
        super(ResidualOptimizationBlock, self).__init__()
        if hidden_dim is None:
            hidden_dim = dimensions
            
        self.residual_mapping = nn.Sequential(
            nn.Linear(dimensions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dimensions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_x = self.residual_mapping(x)
        return f_x + x


def optimize_step(
    model: nn.Module,
    input_data: torch.Tensor,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module
) -> float:
    """Single optimization step leveraging residual formulation.
    
    Uses standard backprop on H(x) = F(x) + x.
    """
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()


# ===================================================================
# TRANSFORMER POSITIONAL EMBEDDINGS (canonical consolidated version)
# Absolute, Relative (bias), and Rotary (RoPE). No changes from previous library.
# ===================================================================

def create_absolute_positional_embeddings(seq_len: int, d_model: int) -> torch.Tensor:
    """Create fixed sinusoidal absolute positional embeddings."""
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def apply_absolute_position_to_embeddings(x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    """Add absolute positional embeddings to token embeddings (input-level)."""
    return x + pe.unsqueeze(0)


def create_relative_index_map(seq_len: int, max_distance: int = 32) -> torch.Tensor:
    """Create clipped relative position index map (pairwise m-n)."""
    pos = torch.arange(seq_len)
    relative_distances = pos.unsqueeze(0) - pos.unsqueeze(1)
    clipped = torch.clamp(relative_distances, -max_distance, max_distance)
    relative_index = clipped + max_distance
    return relative_index


def generate_relative_bias_table(max_distance: int = 32, num_heads: int = 8) -> torch.Tensor:
    """Learnable relative position bias table."""
    num_buckets = 2 * max_distance + 1
    return nn.Parameter(torch.randn(num_heads, num_buckets))


def compute_relative_attention_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    relative_bias_table: torch.Tensor,
    relative_index_map: torch.Tensor
) -> torch.Tensor:
    """Scaled dot-product + relative bias (T5/DeBERTa style)."""
    d_k = q.size(-1)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    seq_len = q.size(-2)
    rel_bias = relative_bias_table[:, relative_index_map.view(-1)]
    rel_bias = rel_bias.view(-1, seq_len, seq_len)
    attn_scores = attn_scores + rel_bias.unsqueeze(0)
    return attn_scores


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the dimensions for RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def generate_rope_embeddings(
    seq_len: int, head_dim: int, theta_base: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate cos/sin for RoPE (geometric decay)."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE rotation (Llama/Mistral style)."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ===================================================================
# SELF-ATTENTION PRELIMINARIES
# Query/Key/Value projections + scaled dot-product attention.
# Uses canonical absolute positional embeddings from above (duplicate removed).
# ===================================================================

def compute_qkv(x: torch.Tensor, d_model: int, d_k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Q, K, V via linear projections (demonstration version).
    
    In production models these would be nn.Linear layers.
    """
    torch.manual_seed(42)
    W_q = torch.randn(d_model, d_k)
    W_k = torch.randn(d_model, d_k)
    W_v = torch.randn(d_model, d_k)
    q = torch.matmul(x, W_q)
    k = torch.matmul(x, W_k)
    v = torch.matmul(x, W_v)
    return q, k, v


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Core scaled dot-product self-attention (Transformer standard)."""
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)
    return output


# ===================================================================
# TRANSFORMER ENCODER-DECODER FLOW
# Demonstrates memory passing between encoder and decoder.
# ===================================================================

class TransformerInteraction(nn.Module):
    """Full encoder-decoder interaction using nn.Transformer.
    
    Inputs: src_tokens, tgt_tokens → output_logits.
    """
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6, vocab_size: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.out(output)