import numpy as np
import matplotlib.pyplot as plt

# Generate dataset: y = x * sin(x) + noise
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = X * np.sin(X) + np.random.randn(80, 1) * 0.5

# Function to map features to polynomial terms
def poly_features(X, degree):
    return np.hstack([X ** i for i in range(degree + 1)])

# Function to normalize features
def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu) / sigma, mu, sigma

# Regularized cost function
def compute_cost(theta, X, y, lambda_):
    m = len(y)
    predictions = X @ theta
    error = predictions - y
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    reg_term = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    return cost + reg_term

# Gradient descent with regularization
def gradient_descent(X, y, theta, alpha, num_iters, lambda_):
    m = len(y)
    for _ in range(num_iters):
        error = X @ theta - y
        grad = (1 / m) * (X.T @ error)
        grad[1:] += (lambda_ / m) * theta[1:]  # Regularize from j=1
        theta -= alpha * grad
    return theta

# Learning curve computation
def learning_curve(X, y, Xval, yval, lambda_):
    m = len(y)
    train_error = []
    val_error = []

    for i in range(1, m + 1):
        X_train = X[:i]
        y_train = y[:i]
        theta = np.zeros((X.shape[1], 1))
        theta = gradient_descent(X_train, y_train, theta, 0.01, 1000, lambda_)
        train_error.append(compute_cost(theta, X_train, y_train, 0))
        val_error.append(compute_cost(theta, Xval, yval, 0))

    return train_error, val_error

# Settings
degree = 5
lambda_ = 1.0

# Prepare features
X_poly = poly_features(X, degree)
X_poly, mu, sigma = feature_normalize(X_poly)
X_poly = np.hstack([np.ones((X_poly.shape[0], 1)), X_poly])
y = y.reshape(-1, 1)

# Split into training and cross-validation sets
split = int(0.6 * len(X))
X_train, X_val = X_poly[:split], X_poly[split:]
y_train, y_val = y[:split], y[split:]

# Train model
theta_init = np.zeros((X_train.shape[1], 1))
theta = gradient_descent(X_train, y_train, theta_init, 0.01, 1000, lambda_)

# Predict for plotting
X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
X_plot_poly = poly_features(X_plot, degree)
X_plot_poly = (X_plot_poly - mu) / sigma
X_plot_poly = np.hstack([np.ones((X_plot_poly.shape[0], 1)), X_plot_poly])
y_plot_pred = X_plot_poly @ theta

# Plot polynomial regression fit
plt.figure(figsize=(7, 5))
plt.scatter(X, y, label="Training data")
plt.plot(X_plot, y_plot_pred, color='red', label="Polynomial fit")
plt.title("Polynomial Regression Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# Plot learning curves
train_error, val_error = learning_curve(X_train, y_train, X_val, y_val, lambda_)
plt.figure(figsize=(7, 5))
plt.plot(range(1, len(train_error) + 1), train_error, label="Training error")
plt.plot(range(1, len(val_error) + 1), val_error, label="Validation error")
plt.title("Learning Curves")
plt.xlabel("Training set size")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.show()

# Print learned theta
print("Learned parameters (theta):\n", theta.flatten())
