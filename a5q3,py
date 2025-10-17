import numpy as np
import matplotlib.pyplot as plt

# Generate linearly separable exam scores and admission results
np.random.seed(0)
X = np.random.randn(100, 2) * 10 + 50  # Exam 1 and Exam 2 scores
y = (X[:, 0] + X[:, 1] > 100).astype(int)  # Admission if sum > 100

# Feature scaling (standardization)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

# Add intercept term
m = len(y)
Xb = np.c_[np.ones(m), X_norm]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = -(1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    for i in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1 / m) * X.T @ (h - y)
        theta = theta - alpha * gradient
        if i % 500 == 0:
            print(f"Iteration {i}: Cost {compute_cost(X, y, theta)}")
    return theta

theta = np.zeros(Xb.shape[1])
alpha = 0.05
iterations = 5000
theta = gradient_descent(Xb, y, theta, alpha, iterations)
print("Learned theta:", theta)

# Plot data points
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='red', label='Not admitted')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', label='Admitted')

# Plot decision boundary
x_vals = np.linspace(np.min(X_norm[:, 0]) - 1, np.max(X_norm[:, 0]) + 1, 100)
y_vals = -(theta[0] + theta[1] * x_vals) / theta[2]

# Convert normalized values back to original for plot axes
x_vals_orig = x_vals * X_std[0] + X_mean[0]
y_vals_orig = y_vals * X_std[1] + X_mean[1]

plt.plot(x_vals_orig, y_vals_orig, 'g-', label='Decision Boundary')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()

# Evaluate accuracy
preds = sigmoid(Xb @ theta) >= 0.5
accuracy = np.mean(preds == y) * 100
print(f'Accuracy on training set: {accuracy:.2f}%')
