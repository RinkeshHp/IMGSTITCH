import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load your CSV data
data = pd.read_csv('CODE/refs/7xaug.csv')
X = data[['x']].values  # Assuming single feature for simplicity
y = data['y'].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to add Gaussian noise to data
def add_noise(X, y, noise_level=0.05):
    noise_X = X + noise_level * np.random.normal(size=X.shape)
    noise_y = y + noise_level * np.random.normal(size=y.shape)
    return noise_X, noise_y

# Function to perform bootstrap sampling
def bootstrap_sampling(X, y, n_samples):
    indices = np.random.choice(range(len(X)), size=n_samples, replace=True)
    return X[indices], y[indices]

# Add Gaussian noise to training data
X_train_noise, y_train_noise = add_noise(X_train, y_train)

# Generate synthetic data via bootstrap sampling
X_train_bootstrap, y_train_bootstrap = bootstrap_sampling(X_train, y_train, 10*len(X_train))

# Combine original, noisy, and bootstrap data
X_train_augmented = np.vstack([X_train, X_train_noise, X_train_bootstrap])
y_train_augmented = np.hstack([y_train, y_train_noise, y_train_bootstrap])

# Save augmented data to a new CSV file
augmented_data = np.hstack([X_train_augmented, y_train_augmented.reshape(-1, 1)])
augmented_df = pd.DataFrame(augmented_data, columns=['x', 'y'])
augmented_df.to_csv('CODE/refs/7xaug.csv', index=False)

# Train Linear Regression model on augmented data
model = LinearRegression()
model.fit(X_train_augmented, y_train_augmented)

# Evaluate the model
y_pred = model.predict(X_test)

# Print model coefficients
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Original Data', color='blue')
plt.scatter(X_train_noise, y_train_noise, label='Noisy Data', color='orange', alpha=0.5)
plt.scatter(X_train_bootstrap, y_train_bootstrap, label='Bootstrap Data', color='green', alpha=0.5)
plt.plot(X_test, y_pred, label='Fitted Line', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Data Augmentation for Linear Regression')
plt.show()
