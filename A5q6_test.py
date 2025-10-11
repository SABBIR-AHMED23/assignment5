import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load digits dataset
digits = load_digits()
X = digits.data  # shape: (1797, 64) — 64 features (8x8 images)
y = digits.target  # labels: 0-9

# Visualize some samples
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray',alpha = 1)
    ax.set_title(f'Label: {digits.target[i]}')
    ax.axis('off')
plt.suptitle('Sample Digits')
plt.show()

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost and gradient for logistic regression
def cost_function(theta, X, y, reg_lambda):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5  # for numerical stability
    cost = (-y @ np.log(h + epsilon) - (1 - y) @ np.log(1 - h + epsilon)) / m
    reg = (reg_lambda / (2 * m)) * np.sum(theta[1:] ** 2)
    return cost + reg

def gradient(theta, X, y, reg_lambda):
    m = len(y)
    h = sigmoid(X @ theta)
    grad = (X.T @ (h - y)) / m
    grad[1:] += (reg_lambda / m) * theta[1:]
    return grad

# Add intercept term
X = np.insert(X, 0, 1, axis=1)  # shape: (1797, 65)

# One-vs-All Training
def one_vs_all(X, y, num_labels, reg_lambda=0.1):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n))
    
    from scipy.optimize import minimize
    for c in range(num_labels):
        y_c = (y == c).astype(int)  # binary for class c
        theta0 = np.zeros(n)
        res = minimize(fun=cost_function, x0=theta0, args=(X, y_c, reg_lambda), 
                       jac=gradient, method='TNC', options={'maxiter': 300})
        all_theta[c] = res.x
    return all_theta

# Prediction
def predict_one_vs_all(all_theta, X):
    probs = sigmoid(X @ all_theta.T)  # shape: (m, num_labels)
    return np.argmax(probs, axis=1)

# Train the OvA logistic regression
num_labels = 10  # digits 0-9
reg_lambda = 0.1
all_theta = one_vs_all(X, y, num_labels, reg_lambda)

# Predict and compute accuracy
y_pred = predict_one_vs_all(all_theta, X)
accuracy = accuracy_score(y, y_pred)

print("Training Accuracy (OvA):", accuracy * 100, "%")

# Display learned theta for each class (first 10 coefficients)
for i in range(num_labels):
    print(f"Class {i} θ (first 10):", all_theta[i][:10])

# Built-in Logistic Regression for comparison
clf = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=300)
clf.fit(digits.data, digits.target)
sk_accuracy = clf.score(digits.data, digits.target)
print("Scikit-learn OvA Accuracy:", sk_accuracy * 100, "%")
