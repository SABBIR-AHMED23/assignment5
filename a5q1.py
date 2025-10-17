import numpy as np

# Sample data from assignment
X = np.array([6.1101, 5.5277, 8.5186])
y = np.array([17.592, 9.1302, 13.662])
m = len(y)

# Add intercept term to X
Xb = np.c_[np.ones(m), X]

# Initialize fitting parameters
theta = np.zeros(2)

# Gradient descent settings
alpha = 0.01
iterations = 1500

def compute_cost(X, y, theta):
    errors = X @ theta - y
    return (1 / (2 * m)) * np.dot(errors, errors)

def gradient_descent(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        gradient = (1 / m) * X.T @ (X @ theta - y)
        theta = theta - alpha * gradient
    return theta

theta = gradient_descent(Xb, y, theta, alpha, iterations)
print("Learned theta:", theta)
