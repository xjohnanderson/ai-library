# self_attention_preliminaries.py
# This script implements the core of self-attention preliminaries in Transformer architecture.
# It shows how query, key, and value vectors are computed from input token embeddings,
# how positional information is integrated, and demonstrates the scaled dot-product attention
# mechanism that produces contextual output representations. The implementation highlights
# the standard linear projections and the quadratic complexity of traditional self-attention.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def compute_qkv(x: torch.Tensor, d_model: int, d_k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # This function computes the query, key, and value vectors using linear projections.
    # It represents the transformation functions f_q, f_k, f_v applied to input embeddings.
    # In practice, three separate linear layers (or one combined) project the embeddings
    # into Q, K, V spaces before positional information is incorporated.
    # Inputs:
    #   x: torch.Tensor - input embeddings of shape [batch, seq_len, d_model]
    #   d_model: int - model dimension (embedding size)
    #   d_k: int - dimension per attention head for Q, K, V
    # Outputs:
    #   Tuple[torch.Tensor, torch.Tensor, torch.Tensor] - (q, k, v) each of shape [batch, seq_len, d_k]
    
    # Create linear projection layers (in a real model these would be nn.Linear modules)
    # For demonstration we use simple matrix multiplication with random weights
    torch.manual_seed(42)
    W_q = torch.randn(d_model, d_k)
    W_k = torch.randn(d_model, d_k)
    W_v = torch.randn(d_model, d_k)
    
    # Apply linear transformations: q = x @ W^Q, k = x @ W^K, v = x @ W^V
    q = torch.matmul(x, W_q)
    k = torch.matmul(x, W_k)
    v = torch.matmul(x, W_v)
    
    return q, k, v


def add_positional_encoding(x: torch.Tensor) -> torch.Tensor:
    # This function adds absolute positional encodings to the input embeddings.
    # This is the traditional way positional information is incorporated before
    # computing query, key, and value vectors (as opposed to RoPE which applies
    # rotations after projection).
    # Inputs:
    #   x: torch.Tensor - input tensor [batch, seq_len, d_model]
    # Outputs:
    #   torch.Tensor - embeddings with added positional encodings, same shape as input
    
    batch_size, seq_len, d_model = x.shape
    # Create absolute positional encodings (sin/cos based)
    position = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # Add positional encodings to embeddings (broadcast over batch)
    x_with_pos = x + pe.unsqueeze(0)
    return x_with_pos


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # This function implements the core self-attention computation.
    # It calculates attention weights a_{m,n} using scaled dot-product of query and key,
    # then produces output o_m as weighted sum of value vectors.
    # This is the standard mechanism used in Transformer self-attention layers.
    # Inputs:
    #   q: torch.Tensor - query tensor [batch, seq_len, d_k] or [batch, heads, seq_len, head_dim]
    #   k: torch.Tensor - key tensor, same shape as q
    #   v: torch.Tensor - value tensor, same shape as q
    #   mask: Optional[torch.Tensor] - optional attention mask
    # Outputs:
    #   torch.Tensor - attention output of shape same as v
    
    d_k = q.size(-1)
    
    # Compute scaled dot-product scores: (q_m^T k_n) / sqrt(d)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided (for causal or padding masking)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Compute attention weights using softmax
    weights = F.softmax(scores, dim=-1)
    
    # Compute weighted sum of values: sum(a_{m,n} * v_n)
    output = torch.matmul(weights, v)
    return output


def demonstrate_self_attention_preliminaries(batch_size: int = 2, seq_len: int = 16, d_model: int = 512, num_heads: int = 8) -> None:
    # This function demonstrates the complete process of computing query, key, value vectors
    # and performing self-attention as described in the preliminaries of Transformer architecture.
    # It shows the flow from input embeddings → positional encoding → QKV projections → attention.
    # Inputs:
    #   batch_size: int - batch size
    #   seq_len: int - sequence length
    #   d_model: int - model embedding dimension
    #   num_heads: int - number of attention heads (for illustration)
    # Outputs:
    #   None - prints explanation and key results
    
    print("=== Self-Attention Preliminaries Demonstration ===")
    print("Core Process:")
    print("1. Input token embeddings x_i")
    print("2. Add absolute positional encodings")
    print("3. Compute query q_m = f_q(x_m + p_m), key k_n = f_k(x_n + p_n), value v_n = f_v(x_n + p_n)")
    print("4. Calculate scaled dot-product attention weights")
    print("5. Output o_m = sum over n of a_{m,n} * v_n")
    
    # Create dummy input embeddings
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Step 1: Add positional encodings (traditional approach)
    x_pos = add_positional_encoding(x)
    
    # Step 2: Compute Q, K, V (head_dim = d_model for simplicity in this demo)
    head_dim = d_model // num_heads
    q, k, v = compute_qkv(x_pos, d_model, d_model)
    
    # Reshape for multi-head (simplified - in real code we split heads)
    # For demonstration we treat as single head with full dimension
    print(f"\nInput shape: {x.shape}")
    print(f"After positional encoding: {x_pos.shape}")
    print(f"Query/Key/Value shape: {q.shape}")
    
    # Step 3: Apply self-attention
    output = scaled_dot_product_attention(q, k, v)
    
    print(f"Self-attention output shape: {output.shape}")
    print("\nKey Properties:")
    print("- Computational complexity of attention matrix is O(N^2)")
    print("- Query, key, value vectors incorporate positional information")
    print("- Attention weights determine contextual importance of each token")
    print("- This quadratic scaling motivates linear attention alternatives and RoPE enhancements")


if __name__ == "__main__":
    # Run the demonstration of self-attention preliminaries
    demonstrate_self_attention_preliminaries(seq_len=32)