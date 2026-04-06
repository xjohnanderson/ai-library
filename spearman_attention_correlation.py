# spearman_attention_correlation.py

import numpy as np
from scipy.stats import spearmanr

# Example: attention weights for a single head (normalized)
# shape: (seq_len, seq_len)
attention_matrix = np.array([[0.1, 0.7, 0.2], 
                             [0.3, 0.3, 0.4], 
                             [0.1, 0.1, 0.8]])

# Example: negative dependency distances derived from a parse tree
# shape: (seq_len, seq_len)
neg_dep_dist = np.array([[ 0, -1, -2],
                         [-1,  0, -1],
                         [-2, -1,  0]])

# Flatten matrices to vectors for correlation calculation
weights_flat = attention_matrix.flatten()
dist_flat = neg_dep_dist.flatten()

# Calculate Spearman Correlation
rho, p_value = spearmanr(weights_flat, dist_flat)

print(f"Spearman rho: {rho:.4f}")