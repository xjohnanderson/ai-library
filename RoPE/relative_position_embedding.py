# relative_position_embedding.py
# This script implements the core concepts of Relative Position Embeddings in Transformer architectures.
# It demonstrates how relative position embeddings differ from absolute ones by focusing on the
# pairwise distance between tokens rather than absolute indices. The implementation shows
# the decomposition of the attention dot-product into content and relative position terms,
# distance clipping, and an efficient relative bias approach for computing attention logits.

import torch
import torch.nn as nn
from typing import Tuple


def create_relative_index_map(seq_len: int, max_distance: int = 32) -> torch.Tensor:
    # This function creates a matrix that maps every pair of positions (m, n) to their
    # clipped relative distance index. This is a key difference from absolute position
    # embeddings which use fixed indices for each token independently.
    # Inputs:
    #   seq_len: int - length of the sequence
    #   max_distance: int - maximum distance to clip relative positions
    # Outputs:
    #   torch.Tensor - relative index map of shape [seq_len, seq_len]
    
    # Create position indices
    pos = torch.arange(seq_len)
    
    # Compute relative distances: m - n for all pairs
    relative_distances = pos.unsqueeze(0) - pos.unsqueeze(1)
    
    # Clip distances to [-max_distance, max_distance] to prevent overfitting to long distances
    clipped = torch.clamp(relative_distances, -max_distance, max_distance)
    
    # Shift to non-negative indices: 0 to 2*max_distance
    relative_index = clipped + max_distance
    
    return relative_index


def generate_relative_bias_table(max_distance: int = 32, num_heads: int = 8) -> torch.Tensor:
    # This function creates a learnable table of relative position biases.
    # Each head can have its own bias for each possible clipped relative distance.
    # This replaces absolute position encodings with relative ones in the attention mechanism.
    # Inputs:
    #   max_distance: int - maximum clipped distance
    #   num_heads: int - number of attention heads
    # Outputs:
    #   torch.Tensor - relative bias table of shape [num_heads, 2*max_distance + 1]
    
    # Total number of unique relative positions after clipping
    num_buckets = 2 * max_distance + 1
    relative_bias_table = nn.Parameter(torch.randn(num_heads, num_buckets))
    
    return relative_bias_table


def compute_relative_attention_scores(q: torch.Tensor, k: torch.Tensor, relative_bias_table: torch.Tensor, relative_index_map: torch.Tensor) -> torch.Tensor:
    # This function computes attention scores with relative position bias.
    # It implements the efficient relative position embedding approach by adding
    # a bias term based on the relative distance between query and key positions.
    # This differs from absolute embeddings by making attention depend on m-n.
    # Inputs:
    #   q: torch.Tensor - query tensor [batch, num_heads, seq_len, head_dim]
    #   k: torch.Tensor - key tensor [batch, num_heads, seq_len, head_dim]
    #   relative_bias_table: torch.Tensor - learnable relative biases
    #   relative_index_map: torch.Tensor - precomputed relative distance indices
    # Outputs:
    #   torch.Tensor - attention scores with relative bias [batch, num_heads, seq_len, seq_len]
    
    # Standard scaled dot-product (content-to-content term)
    d_k = q.size(-1)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    
    # Extract relative biases for each (m, n) pair
    # relative_index_map is [seq_len, seq_len], we gather from the bias table
    seq_len = q.size(-2)
    rel_bias = relative_bias_table[:, relative_index_map.view(-1)]  # [num_heads, seq_len*seq_len]
    rel_bias = rel_bias.view(-1, seq_len, seq_len)  # [num_heads, seq_len, seq_len]
    
    # Add relative bias to attention scores (broadcast over batch)
    attn_scores = attn_scores + rel_bias.unsqueeze(0)
    
    return attn_scores


def demonstrate_relative_vs_absolute(batch_size: int = 2, seq_len: int = 64, num_heads: int = 8, head_dim: int = 64) -> None:
    # This function demonstrates the core differences between relative and absolute
    # position embeddings. It shows how relative embeddings focus on pairwise distances,
    # support better generalization to unseen sequence lengths, and are integrated
    # directly into the attention score computation.
    # Inputs:
    #   batch_size: int - batch size for demonstration
    #   seq_len: int - sequence length
    #   num_heads: int - number of attention heads
    #   head_dim: int - dimension per head
    # Outputs:
    #   None - prints demonstration results and explanations
    
    print("=== Relative Position Embedding vs Absolute Position Embedding ===")
    print("Key Differences:")
    print("• Absolute: Uses fixed position index for each token independently")
    print("• Relative: Uses distance (m - n) between pairs of tokens")
    print("• Relative enables better extrapolation to longer sequences")
    print("• Relative captures local dependencies more naturally")
    
    # Create dummy query and key tensors
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Create relative position components
    max_distance = 32
    relative_index_map = create_relative_index_map(seq_len, max_distance)
    relative_bias_table = generate_relative_bias_table(max_distance, num_heads)
    
    # Compute attention scores with relative bias
    attn_with_relative = compute_relative_attention_scores(q, k, relative_bias_table, relative_index_map)
    
    print(f"\nSequence length: {seq_len}")
    print(f"Relative index map shape: {relative_index_map.shape}")
    print(f"Relative bias table shape: {relative_bias_table.shape}")
    print(f"Attention scores with relative bias shape: {attn_with_relative.shape}")
    
    print("\nCore Advantages of Relative Position Embeddings:")
    print("- Attention depends on relative distance rather than absolute positions")
    print("- Supports generalization to sequence lengths unseen during training")
    print("- Distance clipping prevents overfitting to very long distances")
    print("- Efficient implementations add bias directly to attention logits")
    print("- Often omits unnecessary position-to-position interaction terms")


if __name__ == "__main__":
    # Run the demonstration of relative position embeddings
    demonstrate_relative_vs_absolute(seq_len=128)