import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Dataset
np.random.seed(0)
test1 = np.random.uniform(-1, 1, 100)
test2 = np.random.uniform(-1, 1, 100)

# Circular decision boundary logic (non-linear)
labels = ((test1**2 + test2**2) < 0.5).astype(int)

X = np.column_stack((test1, test2))
y = labels

# 2. Feature Mapping up to 6th degree
def map_features(x1, x2, degree=6):
    m = x1.shape[0]
    output = np.ones((m, 1))
    for i in range(1, degree+1):
        for j in range(i+1):
            term = (x1**(i-j) * x2**j).reshape(m, 1)
            output = np.hstack((output, term))
    return output

# 3. Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 4. Regularized Cost Function
def compute_cost_reg(X, y, theta, lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    reg_term = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    cost = (-y @ np.log(h) - (1 - y) @ np.log(1 - h)) / m + reg_term
    return cost

# 5. Regularized Gradient Descent
def gradient_descent_reg(X, y, theta, alpha, num_iters, lambda_):
    m = len(y)
    cost_history = []

    for _ in range(num_iters):
        h = sigmoid(X @ theta)
        grad = (X.T @ (h-y)) / m
        grad[1:] += (lambda_ / m) * theta[1:]
        theta -= alpha * grad
        cost_history.append(compute_cost_reg(X, y, theta, lambda_))

    return theta, cost_history

# 6. Plot Decision Boundary
def plot_decision_boundary(X_raw, y, theta, lambda_):
    u = np.linspace(-1, 1, 100)
    v = np.linspace(-1, 1, 100)
    z = np.zeros((len(u), len(v)))

    for i in range(len(u)):
        for j in range(len(v)):
            mapped = map_features(np.array([u[i]]), np.array([v[j]]))
            z[i, j] = mapped @ theta

    plt.contour(u, v, z.T, levels=[0], colors='b')
    pos = y == 1
    neg = y == 0
    plt.scatter(X_raw[pos][:, 0], X_raw[pos][:, 1], c='g', marker='+', label='Pass')
    plt.scatter(X_raw[neg][:, 0], X_raw[neg][:, 1], c='r', marker='o', label='Fail')
    plt.title(f"Decision Boundary (lambda={lambda_})")
    plt.legend()
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 7. Training for different λ values
X_mapped = map_features(X[:, 0], X[:, 1])
theta_init = np.zeros(X_mapped.shape[1])
lambda_values = [0, 1, 100]

for lambda_ in lambda_values:
    theta_opt, _ = gradient_descent_reg(X_mapped, y, theta_init.copy(), alpha=1, num_iters=1000, lambda_=lambda_)
    plot_decision_boundary(X, y, theta_opt, lambda_)

# 8. Effect of underfitting vs overfitting
print("""
Discussion:
- λ = 0: No regularization. The model overfits and decision boundary is very complex.
- λ = 1: Balanced regularization. Good generalization with smooth boundary.
- λ = 100: Too much regularization. The model underfits and cannot separate classes well.
""")
