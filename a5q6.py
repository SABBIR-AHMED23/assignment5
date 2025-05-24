import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Create Simulated Digit Data (10 digits, 8x8 pixel images) ---
np.random.seed(0)
num_digits = 10
samples_per_class = 50
image_size = 8 * 8  # 8x8 pixels

# Generate mock digit images and labels
X = np.random.rand(num_digits * samples_per_class, image_size)
y = np.repeat(np.arange(num_digits), samples_per_class)

# Normalize pixel values
X = X / np.max(X)

# Add bias term (intercept)
X = np.hstack([np.ones((X.shape[0], 1)), X])

# --- Step 2: Logistic Regression Functions ---

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = -1/m * (y.T @ np.log(h + epsilon) + (1 - y).T @ np.log(1 - h + epsilon))
    reg = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    return cost + reg

def gradient(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    grad = (1/m) * (X.T @ (h - y))
    grad[1:] += (lambda_ / m) * theta[1:]
    return grad

def gradient_descent(X, y, theta, alpha, num_iters, lambda_):
    for _ in range(num_iters):
        grad = gradient(theta, X, y, lambda_)
        theta -= alpha * grad
    return theta

def one_vs_all(X, y, num_labels, alpha, num_iters, lambda_):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n))
    for c in range(num_labels):
        binary_y = (y == c).astype(int)
        theta = np.zeros(n)
        theta = gradient_descent(X, binary_y, theta, alpha, num_iters, lambda_)
        all_theta[c] = theta
    return all_theta

def predict_one_vs_all(all_theta, X):
    probs = sigmoid(X @ all_theta.T)
    return np.argmax(probs, axis=1)

# --- Step 3: Train and Evaluate Model ---
num_labels = 10
alpha = 0.5
lambda_ = 1.0
num_iters = 300

all_theta = one_vs_all(X, y, num_labels, alpha, num_iters, lambda_)
predictions = predict_one_vs_all(all_theta, X)
accuracy = np.mean(predictions == y) * 100

print(f"Training accuracy: {accuracy:.2f}%")

# --- Step 4: Visualize Some Simulated Digit Images ---
def plot_digits(X_raw, y_raw, num=10):
    plt.figure(figsize=(10, 4))
    for i in range(num):
        digit_img = X_raw[i + 10].reshape(8, 8)
        plt.subplot(2, 5, i + 1)
        plt.imshow(digit_img, cmap='gray')
        plt.title(f"Label: {y_raw[i + 10]}")
        plt.axis('off')
    plt.suptitle("Sample Simulated Digit Images")
    plt.tight_layout()
    plt.show()

# Strip bias term and visualize
plot_digits(X[:, 1:], y)
