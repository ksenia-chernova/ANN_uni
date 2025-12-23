import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

X = np.arange(-20, 20, 0.1).reshape(-1, 1)
Y = np.sin(X)

x_control = np.arange(-20, 20, 0.01).reshape(-1, 1)
y_control = np.sin(x_control)

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
    loss='mse',
    metrics=['mae']
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

control_predictions = model.predict(x_control)
control_mse = np.mean((y_control - control_predictions) ** 2)
control_ss_total = np.sum((y_control - np.mean(y_control)) ** 2)
control_ss_residual = np.sum((y_control - control_predictions) ** 2)
control_r2 = 1 - (control_ss_residual / control_ss_total)

print(f"\nРезультаты на контрольной выборке (4000 точек):")
print(f"Среднеквадратичная ошибка (MSE): {control_mse:.6f}")
print(f"Коэффициент детерминации (R²): {control_r2:.6f}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
x_control_subset = x_control[::4]
y_control_subset = y_control[::4]
control_predictions_subset = control_predictions[::4]

plt.plot(x_control_subset, y_control_subset, 'b-', label='Функция: f(x) = sin(x)', linewidth=2, alpha=0.7)
plt.scatter(x_control_subset, control_predictions_subset, color='red', s=1, label='Предсказания (1000 точек)', alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Сравнение истинной функции и предсказаний')
plt.legend()
plt.grid(True)


plt.subplot(1, 3, 2)
control_errors = y_control_subset - control_predictions_subset
plt.scatter(x_control_subset, control_errors, color='green', s=1, alpha=0.6)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.xlabel('X')
plt.ylabel('Ошибка')
plt.title('Ошибки предсказаний на контрольной выборке')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\nСтатистика ошибок на контрольной выборке (1000 точек):")
print(f"Средняя ошибка: {np.mean(control_errors):.6f}")
print(f"Средняя абсолютная ошибка (MAE): {np.mean(np.abs(control_errors)):.6f}")
print(f"Максимальная ошибка: {np.max(np.abs(control_errors)):.6f}")
print(f"Стандартное отклонение: {np.std(control_errors):.6f}")