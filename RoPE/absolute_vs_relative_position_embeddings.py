# File: absolute_vs_relative_position_embeddings.py
# This script implements and compares the core mechanisms of Absolute Positional Embeddings (APE)
# and Relative Positional Embeddings (RPE), including Rotary Position Embeddings (RoPE) as a
# modern relative-style approach. It demonstrates key architectural differences such as
# input injection point, extrapolation capability, translation invariance, and how
# relative methods modify the attention score directly versus adding to embeddings.

import torch
import math
from typing import Tuple, Optional


def create_absolute_positional_embeddings(seq_len: int, d_model: int) -> torch.Tensor:
    # This function implements Absolute Positional Embeddings (APE) using sinusoidal encodings.
    # It assigns a unique fixed vector to each absolute position index i (0, 1, 2, ...).
    # These embeddings are added directly to token embeddings at the input level.
    # Inputs:
    #   seq_len: int - sequence length
    #   d_model: int - model embedding dimension
    # Outputs:
    #   torch.Tensor - absolute positional embeddings of shape [seq_len, d_model]
    
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe


def create_relative_index_map(seq_len: int, max_relative_distance: int = 32) -> torch.Tensor:
    # This function creates a relative position index map based on pairwise distances (i - j).
    # This is the foundation for Relative Positional Embeddings (RPE), which focus on
    # the distance between tokens rather than their absolute indices.
    # Inputs:
    #   seq_len: int - sequence length
    #   max_relative_distance: int - maximum distance to clip
    # Outputs:
    #   torch.Tensor - relative index map of shape [seq_len, seq_len]
    
    i = torch.arange(seq_len).unsqueeze(1)
    j = torch.arange(seq_len).unsqueeze(0)
    relative_distance = i - j
    
    # Clip to prevent extreme distances
    clipped = torch.clamp(relative_distance, -max_relative_distance, max_relative_distance)
    # Shift to positive indices
    index_map = clipped + max_relative_distance
    
    return index_map


def apply_absolute_position_to_embeddings(x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    # This function demonstrates the Absolute Positional Embedding approach by adding
    # the position vectors directly to the token embeddings before any attention layers.
    # Inputs:
    #   x: torch.Tensor - token embeddings [batch, seq_len, d_model]
    #   pe: torch.Tensor - absolute positional embeddings [seq_len, d_model]
    # Outputs:
    #   torch.Tensor - embeddings with absolute position added, same shape as x
    
    return x + pe.unsqueeze(0)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # This helper function splits the tensor and performs the half-rotation used in RoPE.
    # It is a key component for implementing relative-style positioning via rotations.
    # Inputs:
    #   x: torch.Tensor - input tensor [... , head_dim]
    # Outputs:
    #   torch.Tensor - rotated half tensor
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # This function implements Rotary Position Embeddings (RoPE), a modern relative
    # positional encoding method. It applies rotation matrices to query and key vectors
    # after projection, making the attention inner product depend on relative distance.
    # Inputs:
    #   q: torch.Tensor - query tensor [batch, num_heads, seq_len, head_dim]
    #   k: torch.Tensor - key tensor [batch, num_heads, seq_len, head_dim]
    #   cos: torch.Tensor - cosine values [seq_len, head_dim]
    #   sin: torch.Tensor - sine values [seq_len, head_dim]
    # Outputs:
    #   Tuple[torch.Tensor, torch.Tensor] - rotated (q, k)
    
    # Broadcast cos and sin
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Apply rotation: (x * cos) + (rotate_half(x) * sin)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    
    return q_rot, k_rot


def demonstrate_position_embedding_comparison(batch_size: int = 2, seq_len: int = 64, d_model: int = 512, head_dim: int = 64) -> None:
    # This function demonstrates and compares Absolute Positional Embeddings versus
    # Relative Positional Embeddings (including RoPE). It highlights differences in
    # injection point, extrapolation ability, translation invariance, and computational
    # characteristics as described in the core mechanism comparison.
    # Inputs:
    #   batch_size: int - batch size
    #   seq_len: int - sequence length
    #   d_model: int - model dimension
    #   head_dim: int - head dimension for RoPE demonstration
    # Outputs:
    #   None - prints detailed comparison and results
    
    print("=== Absolute vs Relative Position Embeddings Comparison ===")
    print("Core Mechanism Comparison:")
    print("Absolute (APE): Adds fixed vector to each token embedding at input")
    print("Relative (RPE/RoPE): Injects relative distance information into attention mechanism")
    
    # Create dummy token embeddings
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Absolute Positional Embeddings demonstration
    pe = create_absolute_positional_embeddings(seq_len, d_model)
    x_abs = apply_absolute_position_to_embeddings(x, pe)
    
    print(f"\nAbsolute Positional Embeddings:")
    print(f"  PE shape: {pe.shape}")
    print(f"  Embeddings after adding APE: {x_abs.shape}")
    
    # Relative components demonstration
    relative_index_map = create_relative_index_map(seq_len, max_relative_distance=32)
    
    print(f"\nRelative Positional Embeddings:")
    print(f"  Relative index map shape (pairwise distances): {relative_index_map.shape}")
    print(f"  Relative method modifies attention scores directly (not input embeddings)")
    
    # RoPE demonstration (relative-style)
    num_heads = 8
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Simple cos/sin for RoPE (full implementation would use frequency decay)
    t = torch.arange(seq_len, dtype=torch.float32)
    theta = 10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    freqs = torch.outer(t, theta)
    cos = torch.cos(freqs).repeat(1, 2)  # repeat for full dim
    sin = torch.sin(freqs).repeat(1, 2)
    
    q_rope, k_rope = apply_rope(q, k, cos, sin)
    
    print(f"\nRotary Position Embeddings (RoPE - relative style):")
    print(f"  Query/Key after RoPE: {q_rope.shape}")
    print(f"  Makes attention depend on relative distance via rotation matrices")
    
    print("\nKey Architectural Differences:")
    print("Feature                  | Absolute (APE)          | Relative (RPE / RoPE)")
    print("-------------------------|-------------------------|----------------------")
    print("Injection Point          | Bottom of stack (input) | Inside attention layer")
    print("Extrapolation            | Poor                    | Superior")
    print("Translation Invariance   | Low                     | High")
    print("Computational Cost       | Minimal                 | Higher (per head)")
    print("\nEngineering Trade-offs:")
    print("- APE: Simple, memory efficient, but limited extrapolation")
    print("- RPE/RoPE: Better for long context, syntax, and generalization")


if __name__ == "__main__":
    # Run the comparison demonstration of absolute and relative position embeddings
    demonstrate_position_embedding_comparison(seq_len=128)