# attention_entropy.py

import torch

def calculate_attention_entropy(weights):
    # weights shape: (seq_len, seq_len)
    # Lower entropy = higher structural focus
    entropy = -torch.sum(weights * torch.log(weights + 1e-9), dim=-1)
    return entropy.mean()

# Simulation of the visual shift
baseline_weights = torch.softmax(torch.randn(10, 10), dim=-1)
dasa_weights = torch.softmax(torch.randn(10, 10) + torch.eye(10) * 5, dim=-1)

print(f"Baseline Entropy: {calculate_attention_entropy(baseline_weights):.4f}")
print(f"DASA Entropy: {calculate_attention_entropy(dasa_weights):.4f}")