import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

X = np.arange(-20, 20, 0.1).reshape(-1, 1)
Y = X

np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)
test_size = int(0.2 * len(X))
test_indices = indices[:test_size]
train_indices = indices[test_size:]
X_train = X[train_indices]
X_test = X[test_indices]
Y_train = Y[train_indices]
Y_test = Y[test_indices]

model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',         # Функция потерь: среднеквадратичная ошибка
    metrics=['mae']     # Метрика: средняя абсолютная ошибка
)

history = model.fit(
    X_train, Y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

train_predictions = model.predict(X_train)
train_mse = np.mean((Y_train - train_predictions) ** 2)
train_ss_total = np.sum((Y_train - np.mean(Y_train)) ** 2)
train_ss_residual = np.sum((Y_train - train_predictions) ** 2)
train_r2 = 1 - (train_ss_residual / train_ss_total)

print(f"\nРезультаты на обучающей выборке:")
print(f"Среднеквадратичная ошибка (MSE): {train_mse:.6f}")
print(f"Коэффициент детерминации (R²): {train_r2:.6f}")

test_predictions = model.predict(X_test)
test_mse = np.mean((Y_test - test_predictions) ** 2)
test_ss_total = np.sum((Y_test - np.mean(Y_test)) ** 2)
test_ss_residual = np.sum((Y_test - test_predictions) ** 2)
test_r2 = 1 - (test_ss_residual / test_ss_total)

print(f"\nРезультаты на тестовой выборке:")
print(f"Среднеквадратичная ошибка (MSE): {test_mse:.6f}")
print(f"Коэффициент детерминации (R²): {test_r2:.6f}")

plt.figure(figsize=(16, 8))

plt.subplot(1, 3, 1)
sorted_indices = np.argsort(X_test[:, 0])
X_test_sorted = X_test[sorted_indices]
Y_test_sorted = Y_test[sorted_indices]
predictions_sorted = test_predictions[sorted_indices]

x_range = np.linspace(-20, 20, 400)
y_true = x_range

plt.plot(x_range, y_true, 'b-', label='Истинная функция: f(x) = x', linewidth=2, alpha=0.7)
plt.scatter(X_test_sorted, predictions_sorted, color='red', s=10, label='Предсказания нейронной сети', alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Сравнение функции и предсказаний')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
errors = Y_test_sorted - predictions_sorted
plt.scatter(X_test_sorted, errors, color='green', s=10, alpha=0.6)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.xlabel('X')
plt.ylabel('Ошибка')
plt.title('Ошибки предсказаний')
plt.grid(True)

plt.tight_layout()
plt.show()