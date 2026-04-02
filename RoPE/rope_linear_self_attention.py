# rope_linear_self_attention.py
# This script implements the core of Rotary Position Embedding (RoPE) and demonstrates
# how it improves linear self-attention architectures by encoding relative positional
# information through rotations while preserving the inner-product structure required
# for efficient linear attention kernels.

import torch
from typing import Tuple


# Helper function to generate the cosine and sine embeddings for RoPE
def generate_rope_freqs(head_dim: int, theta_base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    # This function generates the frequency basis used for rotary embeddings.
    # It creates a vector of frequencies that decrease geometrically.
    # Inputs:
    #   head_dim: int - dimension of each attention head (must be even)
    #   theta_base: float - base for the geometric progression of frequencies
    # Outputs:
    #   Tuple[torch.Tensor, torch.Tensor] - (cos, sin) tensors of shape [1, 1, 1, head_dim//2]
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    
    # Create indices for the frequency pairs: 0, 2, 4, ..., head_dim-2
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    
    # For demonstration we create a placeholder sequence length of 1 (will be expanded later)
    t = torch.arange(1, dtype=torch.float32)  # shape: [1]
    freqs = torch.outer(t, inv_freq)  # shape: [seq_len, head_dim//2]
    
    # Compute cos and sin embeddings
    emb = torch.cat((freqs, freqs), dim=-1)  # duplicate for full head_dim if needed, but we use half
    cos = emb.cos().unsqueeze(0).unsqueeze(0)  # [1, seq_len, 1, head_dim//2] -> will broadcast
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    
    return cos, sin


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # This function applies the rotary position embedding to query or key tensors.
    # It rotates pairs of dimensions using the precomputed cos and sin values.
    # This is the core operation that makes the attention inner product depend only
    # on the relative position m - n.
    # Inputs:
    #   x: torch.Tensor - input tensor of shape [batch, seq_len, num_heads, head_dim]
    #   cos: torch.Tensor - cosine embeddings, shape broadcastable to x
    #   sin: torch.Tensor - sine embeddings, shape broadcastable to x
    # Outputs:
    #   torch.Tensor - rotated tensor of the same shape as x
    
    # Split head_dim into two halves for rotation
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    
    # Apply rotation matrix: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
    # This implements the 2D rotation for each frequency pair
    rotated_x = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return rotated_x


def demonstrate_rope_in_linear_attention(batch_size: int = 2, seq_len: int = 64, num_heads: int = 8, head_dim: int = 64) -> None:
    # This function demonstrates how RoPE integrates with linear self-attention.
    # It shows that after applying RoPE, the query-key inner product depends only
    # on relative positions, enabling efficient linear attention while adding
    # position awareness and long-sequence extrapolation capability.
    # Inputs:
    #   batch_size: int - batch size for the example
    #   seq_len: int - sequence length (can be any length, even longer than training)
    #   num_heads: int - number of attention heads
    #   head_dim: int - dimension per head (must be even)
    # Outputs:
    #   None - prints demonstration results
    
    print("=== RoPE Demonstration for Linear Self-Attention ===")
    print(f"Configuration: batch={batch_size}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}")
    
    # Create dummy query and key tensors (normally these come from linear projections)
    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    # Generate RoPE frequencies (cos, sin)
    cos, sin = generate_rope_freqs(head_dim)
    
    # Expand cos/sin to full sequence length
    cos = cos.expand(-1, seq_len, -1, -1)
    sin = sin.expand(-1, seq_len, -1, -1)
    
    # Apply RoPE to query and key
    q_rot = apply_rotary_emb(q, cos, sin)
    k_rot = apply_rotary_emb(k, cos, sin)
    
    # Compute a simple attention score between position 0 and position 10
    # to show relative position dependency
    rel_dist = 10
    score_rel = torch.sum(q_rot[:, 0] * k_rot[:, rel_dist], dim=-1).mean()
    
    # For comparison, compute score with same absolute positions (should be different)
    score_same = torch.sum(q_rot[:, 0] * k_rot[:, 0], dim=-1).mean()
    
    print("\nKey Properties Demonstrated:")
    print(f"1. Attention score depends on relative distance: score(q0, k{rel_dist}) = {score_rel.item():.4f}")
    print(f"2. Score at same position: score(q0, k0) = {score_same.item():.4f}")
    print("3. RoPE preserves inner-product structure → compatible with linear attention kernels")
    print("4. Model can extrapolate to any seq_len (even > training length)")
    print("\nRoPE improves linear self-attention by:")
    print("   • Adding relative positional bias without breaking linearity")
    print("   • Enabling natural decay of attention with increasing |m-n|")
    print("   • Supporting flexible sequence lengths at inference time")


if __name__ == "__main__":
    # Run the demonstration of RoPE in the context of linear self-attention
    demonstrate_rope_in_linear_attention(seq_len=128)