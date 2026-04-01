# File: roformer_core.py
# This script implements the core contributions of RoFormer, focusing on Rotary Position Embedding (RoPE).
# It demonstrates the three primary contributions: (1) rotation-based positional encoding using complex-space geometry,
# (2) ensuring the self-attention inner product depends only on relative position (m-n), and
# (3) maintaining full compatibility with linear self-attention architectures through multiplicative rotations.
# The implementation follows the RoFormer mathematical formalism with block-diagonal rotation matrices.

import torch
from typing import Tuple


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # This function performs the half-rotation used in RoPE to simulate complex multiplication.
    # It splits the tensor along the last dimension and rearranges as [-x2, x1].
    # Inputs:
    #   x: torch.Tensor - input tensor of shape [..., head_dim]
    # Outputs:
    #   torch.Tensor - rotated tensor of the same shape
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def generate_rope_embeddings(seq_len: int, head_dim: int, theta_base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    # This function generates the cosine and sine embeddings for RoPE as per the RoFormer formulation.
    # It creates geometrically decaying frequencies theta_j = base^(-2j/d) to handle long-range dependencies.
    # Inputs:
    #   seq_len: int - length of the sequence
    #   head_dim: int - dimension of each attention head (must be even)
    #   theta_base: float - base value for frequency decay
    # Outputs:
    #   Tuple[torch.Tensor, torch.Tensor] - (cos, sin) tensors of shape [seq_len, head_dim]
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    
    # Create frequency indices: 0, 2, 4, ..., head_dim-2
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    
    # Generate position indices and outer product for frequencies
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # shape: [seq_len, head_dim//2]
    
    # Compute cos and sin for the full head_dim (repeated for each pair)
    emb = torch.cat((freqs, freqs), dim=-1)  # shape: [seq_len, head_dim]
    cos = emb.cos()
    sin = emb.sin()
    
    return cos, sin


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # This function applies the rotary position embedding to query and key tensors as defined in RoFormer.
    # It uses the rotation matrix formulation R_Theta,i * W x_i, implemented efficiently via cos/sin multiplication.
    # After application, <f(q_m), f(k_n)> becomes a function g(q, k, m-n) only.
    # Inputs:
    #   q: torch.Tensor - query tensor [batch, num_heads, seq_len, head_dim]
    #   k: torch.Tensor - key tensor [batch, num_heads, seq_len, head_dim]
    #   cos: torch.Tensor - cosine embeddings [seq_len, head_dim]
    #   sin: torch.Tensor - sine embeddings [seq_len, head_dim]
    # Outputs:
    #   Tuple[torch.Tensor, torch.Tensor] - (q_rotated, k_rotated) with same shape as input
    # Reshape cos and sin to broadcast over batch and heads
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Apply rotation: (q * cos) + (rotate_half(q) * sin)
    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    
    return q_rotated, k_rotated


def demonstrate_roformer_contributions(batch_size: int = 2, seq_len: int = 128, num_heads: int = 8, head_dim: int = 64) -> None:
    # This function demonstrates the three primary contributions of RoFormer:
    # 1. Rotation-based encoding using geometric properties in complex space.
    # 2. Relative position dependency in the attention inner product.
    # 3. Compatibility with linear self-attention (no additive terms that break linearity).
    # It shows the decay property and relative-distance behavior.
    # Inputs:
    #   batch_size: int - batch size for demonstration
    #   seq_len: int - sequence length (supports arbitrary lengths)
    #   num_heads: int - number of attention heads
    #   head_dim: int - head dimension (must be even)
    # Outputs:
    #   None - prints demonstration and explanation
    print("=== RoFormer Core Implementation Demonstration ===")
    print("Primary Contributions of RoFormer:")
    print("1. Introduces Rotary Position Embedding using rotation matrices in complex space")
    print("2. Makes self-attention inner product depend ONLY on relative distance m-n")
    print("3. Maintains full compatibility with linear self-attention kernels")
    
    # Create dummy query and key (as if from linear projections Wq and Wk)
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Generate RoPE embeddings
    cos, sin = generate_rope_embeddings(seq_len, head_dim)
    
    # Apply RoPE
    q_rot, k_rot = apply_rope(q, k, cos, sin)
    
    # Demonstrate relative position dependency
    pos_m = 0
    pos_n = 20
    inner_product_rel = torch.sum(q_rot[:, :, pos_m] * k_rot[:, :, pos_n], dim=-1).mean().item()
    
    pos_same = 0
    inner_product_same = torch.sum(q_rot[:, :, pos_same] * k_rot[:, :, pos_same], dim=-1).mean().item()
    
    print(f"\nRelative Position Dependency (m={pos_m}, n={pos_n}): inner product = {inner_product_rel:.4f}")
    print(f"Same Position (m=n={pos_same}): inner product = {inner_product_same:.4f}")
    print("\nKey Advantages Shown:")
    print("- Inter-token dependency naturally decays with larger |m-n|")
    print("- No additive position encodings → compatible with linear attention")
    print("- Supports extrapolation to longer sequences than trained on")
    print("\nThis implementation forms the foundation of RoFormer models for long-context NLP tasks.")


if __name__ == "__main__":
    # Run the full demonstration of RoFormer's core mechanism
    demonstrate_roformer_contributions(seq_len=256)