# https://www.kaggle.com/datasets/kundanbedmutha/exam-score-prediction-dataset?resource=download

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.metrics import Precision, Recall
import matplotlib.pyplot as plt

df = pd.read_csv('Exam_Score_Prediction.csv')

threshold = 60
df['passed_exam'] = (df['exam_score'] >= threshold).astype(int)

print(f"Распределение классов:")
print(f"Сдали (passed_exam=1): {df['passed_exam'].sum()} студентов")
print(f"Не сдали (passed_exam=0): {len(df) - df['passed_exam'].sum()} студентов")

X = df[['sleep_hours']].values
y = df['passed_exam'].values

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

def manual_standard_scaler(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) # стандартное отклонение
    scaled = (data - mean) / std
    return scaled, mean, std

X_scaled, X_mean, X_std = manual_standard_scaler(X)

def manual_train_test_split(X, y, test_size=0.2, random_seed=42):
    np.random.seed(random_seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    split_idx = int(len(X) * (1 - test_size))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = manual_train_test_split(X_scaled, y, test_size=0.2)

model = Sequential([
    Input(shape=(1,)),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='relu')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(), Recall()]
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

test_results = model.evaluate(X_test, y_test, verbose=0)
print(f"\nРезультаты на тестовых данных:")
print(f"Потери (loss): {test_results[0]:.4f}")
print(f"Точность (accuracy): {test_results[1]:.4f}")
print(f"Точность (precision): {test_results[2]:.4f}")
print(f"Полнота (recall): {test_results[3]:.4f}")

y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

def manual_confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn

tp, tn, fp, fn = manual_confusion_matrix(y_test, y_pred_binary.flatten())

print(f"True Positive (TP): {tp}")
print(f"True Negative (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")

plt.figure(figsize=(5, 5))

X_test_original = X_test * X_std + X_mean
plt.scatter(X_test_original[y_test == 0], y_test[y_test == 0], 
            alpha=0.5, label='Факт: Не сдал', color='red')
plt.scatter(X_test_original[y_test == 1], y_test[y_test == 1], 
            alpha=0.5, label='Факт: Сдал', color='green')
plt.scatter(X_test_original, y_pred, alpha=0.3, 
            label='Предсказание вероятности', color='blue', s=10)
plt.xlabel('Часы сна')
plt.ylabel('Вероятность сдачи/Факт сдачи')
plt.title('Предсказания модели')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()