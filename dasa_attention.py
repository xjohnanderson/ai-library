import torch
import torch.nn.functional as F
import numpy as np

def compute_dasa_attention(Q, K, V, dist_matrix, lambda_param=0.1):
    """
    Simulates DASA modulation on attention logits.
    dist_matrix: (batch, seq_len, seq_len) - pre-computed dependency distances
    """
    d_k = Q.size(-1)
    
    # Standard Scaled Dot-Product
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    
    # Generate Syntactic Bias (phi): Inverse distance weighting
    # We add 1 to avoid division by zero; high distance = low bias
    syntactic_bias = 1.0 / (dist_matrix.float() + 1.0)
    
    # Modulate logits
    modulated_scores = scores + (lambda_param * syntactic_bias)
    
    # Softmax and context vector generation
    attn_weights = F.softmax(modulated_scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights

# Example usage
seq_len = 5
dim = 64
Q = torch.randn(1, seq_len, dim)
K = torch.randn(1, seq_len, dim)
V = torch.randn(1, seq_len, dim)
# Mock dependency distance matrix (0 on diagonal, 1-4 for relations)
dist = torch.tensor([[[0, 1, 2, 2, 3], 
                      [1, 0, 1, 3, 4], 
                      [2, 1, 0, 1, 2], 
                      [2, 3, 1, 0, 1], 
                      [3, 4, 2, 1, 0]]])

output, weights = compute_dasa_attention(Q, K, V, dist)