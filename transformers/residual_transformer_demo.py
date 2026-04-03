# residual_transformer_demo.py
 

from residual_transformer_core import (
    ResidualBlock,
    ResidualOptimizationBlock,
    optimize_step,
    create_absolute_positional_embeddings,
    apply_absolute_position_to_embeddings,
    compute_qkv,
    scaled_dot_product_attention,
    TransformerInteraction,
    # RoPE / RPE helpers also available if needed
)

import torch
import torch.nn as nn


def demonstrate_cnn_residual_block_for_vision():
    """Demonstrates the CNN-style ResidualBlock (vision/ResNet use-case)."""
    print("\n=== DEMO: CNN Residual Block (Vision / ResNet-style) ===")
    input_tensor = torch.randn(1, 64, 64, 64)  # batch, channels, H, W
    res_block = ResidualBlock(in_channels=64, out_channels=64)
    output = res_block(input_tensor)
    print(f"Input shape:  {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Residual shortcut applied – identity mapping possible.")


def demonstrate_residual_learning_optimization_in_mlp():
    """Demonstrates the general residual hypothesis in an MLP block + optimization step."""
    print("\n=== DEMO: Residual Optimization Block (MLP / General) ===")
    d_in = 16
    block = ResidualOptimizationBlock(dimensions=d_in)
    sample_input = torch.randn(1, d_in)
    sample_target = sample_input.clone()
    
    opt = torch.optim.SGD(block.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    
    loss = optimize_step(block, sample_input, sample_target, opt, loss_fn)
    print(f"Initial optimization loss: {loss:.6f}")
    print("✓ Block can learn identity (F(x)→0) via residual connection.")


def demonstrate_self_attention_preliminaries_with_position_embeddings():
    """Demonstrates self-attention preliminaries composed with canonical absolute PE."""
    print("\n=== DEMO: Self-Attention Preliminaries (with Position Embeddings) ===")
    batch_size, seq_len, d_model = 2, 16, 512
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Use canonical absolute positional embeddings (duplicate removed)
    pe = create_absolute_positional_embeddings(seq_len, d_model)
    x_pos = apply_absolute_position_to_embeddings(x, pe)
    
    head_dim = d_model  # simplified single-head demo
    q, k, v = compute_qkv(x_pos, d_model, head_dim)
    
    output = scaled_dot_product_attention(q, k, v)
    
    print(f"Input shape:               {x.shape}")
    print(f"After absolute PE:         {x_pos.shape}")
    print(f"Q/K/V shape:               {q.shape}")
    print(f"Self-attention output:     {output.shape}")
    print("✓ Quadratic attention + positional integration complete.")


def demonstrate_transformer_encoder_decoder_flow():
    """Demonstrates encoder → decoder memory passing in a full Transformer."""
    print("\n=== DEMO: Transformer Encoder-Decoder Interaction ===")
    batch_size = 2
    src_seq_len, tgt_seq_len, vocab_sz = 10, 8, 1000
    
    model = TransformerInteraction(vocab_size=vocab_sz)
    src_input = torch.randint(0, vocab_sz, (batch_size, src_seq_len))
    tgt_input = torch.randint(0, vocab_sz, (batch_size, tgt_seq_len))
    
    logits = model(src_input, tgt_input)
    
    print(f"Encoder input (src): {src_input.shape}")
    print(f"Decoder input (tgt): {tgt_input.shape}")
    print(f"Output logits:       {logits.shape} (batch, tgt_len, vocab)")
    print("✓ Full encoder-decoder flow with causal masking.")


def run_all_demos():
    """Orchestrates all descriptive demonstrations in one run."""
    print("=== Residual + Transformer Architecture Library Demo ===\n")
    demonstrate_cnn_residual_block_for_vision()
    demonstrate_residual_learning_optimization_in_mlp()
    demonstrate_self_attention_preliminaries_with_position_embeddings()
    demonstrate_transformer_encoder_decoder_flow()
    print("\n✓ All core building blocks demonstrated. Library is ready for composition!")


if __name__ == "__main__":
    run_all_demos()