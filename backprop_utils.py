# backprop_utils.py
# Calculates layer errors and gradients for a standard MLP.
# Inputs: activations (list of arrays), zs (list of weighted inputs), 
#         weights (list of matrices), target (array)
# Outputs: delta_list (list of error terms), gradients (list of weight gradients)

import numpy as np

def compute_error_terms(activations, zs, weights, target):
    # Initialize lists to store errors
    deltas = [None] * len(weights)
    
    # 1. Output error: (a - y) * sigma_prime(z)
    # Assuming Mean Squared Error and Sigmoid activation
    output_activation = activations[-1]
    cost_gradient = output_activation - target
    deltas[-1] = cost_gradient * sigmoid_prime(zs[-1])
    
    # 2. Backpropagate error to hidden layers
    for l in range(len(weights) - 2, -1, -1):
        # Weight matrix l+1 connects layer l to l+1
        deltas[l] = np.dot(weights[l+1].T, deltas[l+1]) * sigmoid_prime(zs[l])
        
    return deltas

def sigmoid_prime(z):
    # Derivative of the sigmoid function: sigma(z) * (1 - sigma(z))
    s = 1 / (1 + np.exp(-z))
    return s * (1 - s)