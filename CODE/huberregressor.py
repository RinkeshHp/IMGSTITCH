import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#load csv data
data = pd.read_csv('CODE/refs/5xaug.csv')
X = data[['x']].values
y2p5 = data['y'].values

#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y2p5, test_size=0.2, random_state=42)

#optionally normalize the data
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()
# X_train_scaled = scaler_X.fit_transform(X_train)
# y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
# X_test_scaled = scaler_X.transform(X_test)
# y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

X_train_scaled = X_train
X_test_scaled = X_test
y_train_scaled = y_train
y_test_scaled = y_test
#train the model with huber loss
huber = HuberRegressor()
huber.fit(X_train_scaled, y_train_scaled)



#Make predictions
y_train_pred_scaled = huber.predict(X_train_scaled)
y_test_pred_scaled = huber.predict(X_test_scaled)



#inverse transform to original scale
# y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
# y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

y_train_pred = y_train_pred_scaled
y_test_pred = y_test_pred_scaled


#evaluate the model
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Train MSE: {mse_train}")
print(f"Test MSE: {mse_test}")
print(f"Train MAE: {mae_train}")
print(f"Test MAE: {mae_test}")
print(f"Train R2: {r2_train}")
print(f"Test R2: {r2_test}")

print(f"coefficients:{huber.coef_}")
print(f"Intercept:{huber.intercept_}")

plt.figure(figsize=(10, 6))
plt.scatter(X, y2p5, label='Data', color='blue')
# plt.text(0.5, 6, f"y={huber.coef_[0]}x+{huber.intercept_}", fontsize=12, ha='center')
plt.plot(X_train, y_train_pred, label='Huber Regressor', color='red')

plt.xlabel('step size(mm)')
plt.ylabel('translation(pixels)')
plt.title(f"Huber Regressor for zoom 5x (y={huber.coef_[0]}x+{huber.intercept_})")
plt.legend()
plt.grid(True)
plt.show()


