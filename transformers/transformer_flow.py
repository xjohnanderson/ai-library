# transformer_flow.py
# This script demonstrates the data interaction between Encoder and Decoder modules.
# Inputs: src_tokens (LongTensor [batch, src_len]), tgt_tokens (LongTensor [batch, tgt_len])
# Outputs: output_logits (FloatTensor [batch, tgt_len, vocab_size])

import torch
import torch.nn as nn

class TransformerInteraction(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, vocab_size=5000):
        super(TransformerInteraction, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # The standard Transformer module handles the interaction internally
        # but we can observe the memory pass here.
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # 1. Encoder Processing: Transforms raw source into context 'memory'
        # Shape: [batch, src_seq_len, d_model]
        src_emb = self.embedding(src)
        
        # 2. Decoder Processing: Uses 'memory' from encoder and shifted target
        # Shape: [batch, tgt_seq_len, d_model]
        tgt_emb = self.embedding(tgt)
        
        # Create a causal mask for the decoder so it can't "see" the future
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # The 'memory' argument is the output of the Encoder
        output = self.transformer(
            src_emb, 
            tgt_emb, 
            tgt_mask=tgt_mask
        )
        
        return self.out(output)

# --- Execution Simulation ---
if __name__ == "__main__":
    # Mock data for GitHub Codespaces environment
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    vocab_sz = 1000

    model = TransformerInteraction(vocab_size=vocab_sz)
    
    # Encoder Input: Full sentence at once
    src_input = torch.randint(0, vocab_sz, (batch_size, src_seq_len))
    
    # Decoder Input: Shifted target (words generated so far)
    tgt_input = torch.randint(0, vocab_sz, (batch_size, tgt_seq_len))

    logits = model(src_input, tgt_input)

    print(f"Encoder Input Shape:  {src_input.shape}")
    print(f"Decoder Input Shape:  {tgt_input.shape}")
    print(f"Output Logits Shape:  {logits.shape} (Batch, Tgt_Len, Vocab)")