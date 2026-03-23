# hyperparameter_tuning.py
# This script implements manual hyperparameter tuning for a simple linear regression model trained via gradient descent. It generates synthetic data, performs a train-validation-test split, conducts a grid search over learning rates as the hyperparameter, trains the model for each candidate value, evaluates mean squared error on the validation set to select the best learning rate, and outputs the selected hyperparameter along with final test performance. The implementation uses only NumPy for minimal dependencies.

import numpy as np

def generate_synthetic_data(n_samples=200, noise=0.1):
    X = np.random.rand(n_samples, 1)
    y = 3 * X + 2 + noise * np.random.randn(n_samples, 1)
    return X, y

def train_linear_regression(X_train, y_train, learning_rate, epochs=100):
    # Augment X with bias term
    Xb = np.c_[np.ones((len(X_train), 1)), X_train]
    theta = np.zeros((2, 1))
    m = len(X_train)
    for _ in range(epochs):
        gradients = (2 / m) * Xb.T.dot(Xb.dot(theta) - y_train)
        theta -= learning_rate * gradients
    return theta

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Main execution
X, y = generate_synthetic_data()
# Simple split: 60% train, 20% val, 20% test
indices = np.random.permutation(len(X))
train_idx = indices[:120]
val_idx = indices[120:160]
test_idx = indices[160:]
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Hyperparameter grid: learning rates
learning_rates = [0.01, 0.1, 0.5, 1.0]
best_lr = None
best_val_mse = float('inf')

for lr in learning_rates:
    theta = train_linear_regression(X_train, y_train, lr)
    # Predict on val
    Xb_val = np.c_[np.ones((len(X_val), 1)), X_val]
    y_pred_val = Xb_val.dot(theta)
    val_mse = mse(y_val, y_pred_val)
    print(f"Learning rate: {lr}, Validation MSE: {val_mse:.4f}")
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_lr = lr

print(f"Best learning rate: {best_lr}")

# Final evaluation on test
theta_final = train_linear_regression(X_train, y_train, best_lr)
Xb_test = np.c_[np.ones((len(X_test), 1)), X_test]
y_pred_test = Xb_test.dot(theta_final)
test_mse = mse(y_test, y_pred_test)
print(f"Test MSE with best hyperparameter: {test_mse:.4f}")