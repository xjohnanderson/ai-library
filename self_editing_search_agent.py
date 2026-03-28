# self_editing_search_agent.py

import numpy as np
from typing import List, Dict
import tiktoken  # For token counting

class ContextManager:
    """
    Implements Chroma's 2026 'Context-1' self-editing logic.
    Bounded context window with semantic pruning.
    """
    def __init__(self, token_budget: int = 4000, model_name: str = "gpt-4o"):
        self.token_budget = token_budget
        self.encoder = tiktoken.encoding_for_model(model_name)
        self.context_pool: List[Dict] = [] # Stores {id, text, score, tokens}

    def add_chunks(self, new_chunks: List[Dict]):
        """Adds chunks and calculates token footprint."""
        for chunk in new_chunks:
            chunk['tokens'] = len(self.encoder.encode(chunk['text']))
            self.context_pool.append(chunk)
        
        # Immediate self-edit if over budget
        if self.current_token_count() > self.token_budget:
            self.prune_context()

    def current_token_count(self) -> int:
        return sum(c['tokens'] for c in self.context_pool)

    def prune_context(self):
        """
        The 'Context-1' Edit: Sorts by semantic utility and discards 
        the bottom 25% or until under budget.
        """
        # Sort by relevance score (provided by retriever or cross-encoder)
        self.context_pool.sort(key=lambda x: x['score'], reverse=True)
        
        while self.current_token_count() > self.token_budget:
            discarded = self.context_pool.pop()
            print(f"[Self-Edit] Discarded low-utility chunk: {discarded['id']}")

    def get_final_payload(self) -> str:
        """Returns the 'Shuffled' haystack for better performance."""
        texts = [c['text'] for c in self.context_pool]
        # Research shows 'shuffled' context often outperforms logical flow
        np.random.shuffle(texts) 
        return "\n\n".join(texts)

# --- Simulation for your AI/Quantum Library components ---
agent = ContextManager(token_budget=1000)

# Hop 1: Basic retrieval
agent.add_chunks([
    {"id": "Q1", "text": "Quantum phase estimation logic...", "score": 0.92},
    {"id": "D1", "text": "Distractor: The history of computing...", "score": 0.45}
])

# Hop 2: Deep search (pushes over budget)
agent.add_chunks([
    {"id": "Q2", "text": "Detailed error correction codes...", "score": 0.88},
    {"id": "M1", "text": "Linear algebra fundamentals...", "score": 0.30}
])

print(f"Final Active Context Size: {agent.current_token_count()} tokens")