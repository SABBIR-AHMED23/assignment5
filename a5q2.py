import numpy as np

# Sample data
X = np.array([[2104, 5], [1600, 3], [2400, 4]])
y = np.array([399900, 329900, 369000])
m = len(y)

# Feature normalization
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

# Add intercept term
Xb = np.c_[np.ones(m), X_norm]

# Initialize fitting parameters
theta = np.zeros(Xb.shape[1])

# Gradient descent settings
alpha = 0.01
iterations = 400

def gradient_descent(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        gradient = (1 / m) * X.T @ (X @ theta - y)
        theta = theta - alpha * gradient
    return theta

theta = gradient_descent(Xb, y, theta, alpha, iterations)
print("Learned theta:", theta)
