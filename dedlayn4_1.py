import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
X, y = make_regression(n_samples=1000, n_features=3, noise=0.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

training_loss = []
validation_loss = []

model = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=1, warm_start=True, random_state=42)

epochs = 100
for epoch in range(epochs):
    model.fit(X_train, y_train)  
    
    y_train_pred = model.predict(X_train)
    train_loss = mean_squared_error(y_train, y_train_pred)
    training_loss.append(train_loss)
    
    y_val_pred = model.predict(X_test)
    val_loss = mean_squared_error(y_test, y_val_pred)
    validation_loss.append(val_loss)
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.title('Model Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (Mean Squared Error)')
plt.legend()
plt.show()
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
plt.scatter(y_test, y_pred)
plt.xlabel('Asl qiymat')
plt.ylabel('Bashorat qilingan qiymat')
plt.title('Asl va Bashorat Qilingan Qiymatlar')
plt.show()

