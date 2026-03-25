# lstm_cell.py
# Minimal NumPy implementation of a single LSTM cell demonstrating all three gating mechanisms.
# Shows explicit computation of forget, input, and output gates with cell state update.

import numpy as np

class LSTMCell:
    def __init__(self, input_size: int, hidden_size: int):
        # Weights for the four gates combined (forget, input, candidate, output)
        self.W = np.random.randn(4 * hidden_size, input_size + hidden_size) * 0.01
        self.b = np.zeros((4 * hidden_size, 1))

    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        """
        x: (input_size, 1)
        h_prev, c_prev: (hidden_size, 1)
        Returns: h_next, c_next
        """
        # Concatenate previous hidden and current input
        combined = np.concatenate((h_prev, x), axis=0)          # (input_size + hidden_size, 1)

        # Linear transformation for all gates at once
        gates = np.dot(self.W, combined) + self.b               # (4*hidden_size, 1)

        # Split into four gates
        f, i, c_tilde, o = np.split(gates, 4, axis=0)

        # Apply activations
        f = 1 / (1 + np.exp(-f))           # forget gate σ
        i = 1 / (1 + np.exp(-i))           # input gate σ
        c_tilde = np.tanh(c_tilde)         # candidate cell
        o = 1 / (1 + np.exp(-o))           # output gate σ

        # Cell state update (core memory highway)
        c_next = f * c_prev + i * c_tilde

        # Hidden state (filtered by output gate)
        h_next = o * np.tanh(c_next)

        return h_next, c_next