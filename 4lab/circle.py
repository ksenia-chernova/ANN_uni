import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

theta = np.linspace(0, 2*np.pi, 400)
X_circle = np.cos(theta).reshape(-1, 1)
Y_circle = np.sin(theta).reshape(-1, 1)

theta_control = np.linspace(0, 2*np.pi, 1000)
X_control = np.cos(theta_control).reshape(-1, 1)
Y_control = np.sin(theta_control).reshape(-1, 1)

np.random.seed(42)
indices = np.arange(400)
np.random.shuffle(indices)
test_size = 80
test_indices = indices[:test_size]
train_indices = indices[test_size:]
X_train = X_circle[train_indices]
X_test = X_circle[test_indices]
Y_train = Y_circle[train_indices]
Y_test = Y_circle[test_indices]

model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
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
y_train_mean = np.mean(Y_train)
train_ss_total = np.sum((Y_train - y_train_mean) ** 2)
train_ss_residual = np.sum((Y_train - train_predictions) ** 2)
train_r2 = 1 - (train_ss_residual / train_ss_total)

print(f"\nРезультаты на обучающей выборке:")
print(f"Среднеквадратичная ошибка (MSE): {train_mse:.6f}")
print(f"Коэффициент детерминации (R²): {train_r2:.6f}")

test_predictions = model.predict(X_test)
test_mse = np.mean((Y_test - test_predictions) ** 2)
y_test_mean = np.mean(Y_test)
test_ss_total = np.sum((Y_test - y_test_mean) ** 2)
test_ss_residual = np.sum((Y_test - test_predictions) ** 2)
test_r2 = 1 - (test_ss_residual / test_ss_total)

print(f"\nРезультаты на тестовой выборке:")
print(f"Среднеквадратичная ошибка (MSE): {test_mse:.6f}")
print(f"Коэффициент детерминации (R²): {test_r2:.6f}")

control_predictions = model.predict(X_control)
control_mse = np.mean((Y_control - control_predictions) ** 2)
y_control_mean = np.mean(Y_control)
control_ss_total = np.sum((Y_control - y_control_mean) ** 2)
control_ss_residual = np.sum((Y_control - control_predictions) ** 2)
control_r2 = 1 - (control_ss_residual / control_ss_total)

print(f"\nРезультаты на контрольной выборке (1000 точек):")
print(f"Среднеквадратичная ошибка (MSE): {control_mse:.6f}")
print(f"Коэффициент детерминации (R²): {control_r2:.6f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(np.cos(theta_control), np.sin(theta_control), 'b-', label='Истинная окружность', linewidth=2, alpha=0.7)
plt.scatter(X_control, control_predictions, color='red', s=1, label='Предсказания (1000 точек)', alpha=0.6)
plt.xlabel('X = cos(θ)')
plt.ylabel('Y = sin(θ)')
plt.title('Сравнение истинной окружности и предсказаний')
plt.axis('equal')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
control_errors = Y_control - control_predictions
plt.plot(theta_control, control_errors, 'g-', linewidth=1, alpha=0.7)
plt.fill_between(theta_control, control_errors[:, 0], 0, alpha=0.2, color='green')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.xlabel('θ (радианы)')
plt.ylabel('Ошибка предсказания')
plt.title('Ошибки предсказаний на контрольной выборке')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\nСтатистика ошибок на контрольной выборке (1000 точек):")
print(f"Средняя ошибка: {np.mean(control_errors):.6f}")
print(f"Средняя абсолютная ошибка (MAE): {np.mean(np.abs(control_errors)):.6f}")
print(f"Максимальная ошибка: {np.max(np.abs(control_errors)):.6f}")
print(f"Стандартное отклонение ошибок: {np.std(control_errors):.6f}")

print(f"\nПроверка радиуса контрольной окружности:")
radii = np.sqrt(X_control**2 + control_predictions**2)
mean_radius = np.mean(radii)
std_radius = np.std(radii)
print(f"Средний радиус предсказаний: {mean_radius:.6f}")
print(f"Стандартное отклонение радиуса: {std_radius:.6f}")