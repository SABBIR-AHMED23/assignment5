import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split 

# Step 1: Generate synthetic non-linear dataset
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = X.ravel() * np.sin(X.ravel()) + np.random.normal(0, 2, X.shape[0])

# Step 2: Polynomial feature transformation
poly_degree = 5
poly = PolynomialFeatures(degree=poly_degree)
X_poly = poly.fit_transform(X)

# Step 3: Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_poly, y, test_size=0.3, random_state=42)

# Step 4: Train model with Ridge regularization
lambda_val = 1.0
model = Ridge(alpha=lambda_val)
model.fit(X_train, y_train)

# Step 5: Predict for plotting
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)

# Print learned parameters
print("Learned Parameters (θ):", model.coef_)

# Step 6: Plot polynomial fit
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_plot, y_plot, color='red', label=f'Degree {poly_degree} Fit (λ={lambda_val})')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression Fit')
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Plot learning curves
def plot_learning_curve(model, X_train, y_train, X_val, y_val, train_sizes):
    train_errors = []
    val_errors = []

    for m in train_sizes:
        model.fit(X_train[:m], y_train[:m])
        train_pred = model.predict(X_train[:m])
        val_pred = model.predict(X_val)

        train_errors.append(mean_squared_error(y_train[:m], train_pred))
        val_errors.append(mean_squared_error(y_val, val_pred))

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_errors, label='Training Error')
    plt.plot(train_sizes, val_errors, label='Validation Error')
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

train_sizes = np.linspace(10, len(X_train), 10, dtype=int)
plot_learning_curve(Ridge(alpha=lambda_val), X_train, y_train, X_val, y_val, train_sizes)

# Step 8: Analyze bias-variance trade-off by varying λ
lambdas = [0, 0.01, 0.1, 1, 10, 100]
train_mse = []
val_mse = []

for l in lambdas:
    model = Ridge(alpha=l)
    model.fit(X_train, y_train)
    train_mse.append(mean_squared_error(y_train, model.predict(X_train)))
    val_mse.append(mean_squared_error(y_val, model.predict(X_val)))

plt.figure(figsize=(8, 5))
plt.plot(lambdas, train_mse, marker='o', label='Training MSE')
plt.plot(lambdas, val_mse, marker='x', label='Validation MSE')
plt.xscale('log')
plt.xlabel('λ (Regularization Parameter)')
plt.ylabel('Mean Squared Error')
plt.title('Bias-Variance Trade-off')
plt.legend()
plt.grid(True)
plt.show()

lambdas = [0, 0.01, 1, 100]
plt.scatter(X,y,color = 'blue',label ='Training Data',alpha=0.5)
for i,l in enumerate(lambdas):
    model = Ridge(alpha=l)
    model.fit(X_train, y_train)
    y_plot = model.predict(X_plot_poly)

    plt.plot(X_plot, y_plot, label=f'λ = {l}')

plt.title('Polynomial Regression for Different λ')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()