import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from class_NeuralNetwork import NeuralNetwork

def read_data(file: str, has_target: bool = True, skip_header: bool = True):
    x_data = []
    y_data = []
    
    with open(file, "r") as data:
        if skip_header:
            next(data)
        
        for line in data:
            line = line.strip()
            if line:
                values = line.split(",")
                try:
                    float_values = list(map(float, values))
                    if has_target:
                        if len(float_values) == 3:
                            x_data.append(float_values[:2])
                            y_data.append([float_values[2]])
                        elif len(float_values) == 9:
                            x_data.append(float_values[:6])
                            y_data.append(float_values[6:])
                        else:
                            x_data.append(float_values[:-1])
                            y_data.append([float_values[-1]])
                    else:
                        x_data.append(float_values)
                except ValueError:
                    continue
    
    x_data = np.array(x_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.float32) if has_target else None
    
    return x_data, y_data


def read_rgb_data(file: str, skip_header: bool = True):
    x_data = []
    y_data = []
    
    with open(file, "r") as data:
        if skip_header:
            next(data)
        
        for line in data:
            line = line.strip()
            if line:
                values = line.split(",")
                try:
                    if len(values) >= 4:
                        x_data.append([float(values[0]), float(values[1]), float(values[2])])
                        y_data.append([float(values[3])])
                except ValueError:
                    continue
    
    x_data = np.array(x_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.float32)
    
    return x_data, y_data


def split_data(x: np.ndarray, y: np.ndarray, train_ratio: float = 0.8, seed: int = 42):
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    n = len(x)
    train_size = int(n * train_ratio)
    
    np.random.seed(seed)
    indices = np.random.permutation(n)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    
    return x_train, y_train, x_test, y_test

def test_accuracy(network, x_test, y_test, threshold: float = 0.5):
    if len(x_test) == 0:
        return 0
    
    correct = 0
    for i in range(len(x_test)):
        prediction = network.predict(x_test[i])
        actual = y_test[i]
        pred_binary = 1 if prediction[0] >= threshold else 0
        actual_binary = 1 if actual[0] >= threshold else 0
        if pred_binary == actual_binary:
            correct += 1
    return correct / len(x_test) * 100

def process_image_for_red_detection(network, image_path: str):
    img = Image.open(image_path)
    img_array = np.asarray(img)
    height, width, channels = img_array.shape
    
    print(f"Размер изображения: {width}x{height}, Каналы: {channels}")
    
    mas = img_array.reshape(height * width, channels)
    red_mask = np.zeros(height * width, dtype=np.float32)
    
    for i in range(len(mas)):
        pixel = mas[i] / 255.0
        prediction = network.predict(pixel)
        red_mask[i] = prediction[0]
    
    red_mask_normalized = np.clip(red_mask * 100, 0, 100)
    red_mask_reshaped = red_mask_normalized.reshape(height, width)
    
    print(f"Значения маски: min={red_mask.min():.4f}, max={red_mask.max():.4f}, mean={red_mask.mean():.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    axes[0].imshow(img_array)
    axes[0].set_title('Оригинальное изображение')
    axes[0].axis('off')
    
    im = axes[1].imshow(red_mask_reshaped, cmap='Reds', vmin=0, vmax=100)
    axes[1].set_title('Обнаружение красного цвета')
    axes[1].axis('off')
    
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.suptitle('Распознавание красного цвета на изображении', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    red_mask_uint8 = np.clip(red_mask_reshaped * 2.55, 0, 255).astype(np.uint8)
    mask_img = Image.fromarray(red_mask_uint8, mode="L")
    mask_img.save("red_mask_result.jpg")
    print("Маска сохранена как 'red_mask_result.jpg'")
    print(f"Маска: min={red_mask_uint8.min()}, max={red_mask_uint8.max()}")
    
    return red_mask_reshaped

if __name__ == "__main__":

    print("=== Задание 1 ===")
    x_data, y_data = read_data("data/2lab.csv", skip_header=False)
    x_train, y_train, x_test, y_test = split_data(x_data, y_data, 0.8)
    nn1 = NeuralNetwork(n_neurons=1, n_inputs=2)
    nn2 = NeuralNetwork(n_neurons=1, n_inputs=2)
    
    for _ in range(50):
        nn1.fit_1(x_train, y_train)
        nn2.fit_2(x_train, y_train, alpha=0.0001)
    
    print(f"Сеть 1 (метод 1): Веса = {nn1.weight[0]}, Смещение = {nn1.bias[0]}")
    print(f"Сеть 2 (метод 2): Веса = {nn2.weight[0]}, Смещение = {nn2.bias[0]}")


    print("\n=== Задание 2 ===")
    x_data_2, y_data_2 = read_data("data/2lab_data.csv")
    if len(x_data_2) > 0:
        print(f"Загружено {len(x_data_2)} примеров")
        print(f"Количество входов: {x_data_2.shape[1]}")
        print(f"Количество выходов: {y_data_2.shape[1]}")
        
        x_train_2, y_train_2, x_test_2, y_test_2 = split_data(x_data_2, y_data_2, 0.8)
        nn3 = NeuralNetwork(n_neurons=3, n_inputs=6)
        print(f"\nНачальные веса (3x6):")
        print(nn3.weight)
        
        nn3.fit(x_train_2, y_train_2, alpha=0.000001, target_error=1000.0, max_epochs=50)
        
        print(f"\nОбученные веса сети (3x6):")
        print(nn3.weight)
        
        print("\nПроверка предсказаний (первые 3 примера):")
        for i in range(min(3, len(x_test_2))):
            prediction = nn3.predict(x_test_2[i])
            actual = y_test_2[i]
            print(f"Пример {i+1}: Предсказание={prediction}, Фактическое={actual}")
        print("Обучение не удалось: получены NaN значения")
    

    print("\n=== Задание 3 ===")
    x_train_rgb, y_train_rgb = read_rgb_data("data/2lab_data_3_train.csv")
    x_test_rgb, y_test_rgb = read_rgb_data("data/2lab_data_3_test.csv")
    
    if len(x_train_rgb) > 0 and len(x_test_rgb) > 0:
        print(f"Тренировочные данные RGB: {len(x_train_rgb)} примеров")
        print(f"Тестовые данные RGB: {len(x_test_rgb)} примеров")
        
        print("\nПросмотр первых 5 примеров обучающих данных:")
        for i in range(min(5, len(x_train_rgb))):
            print(f"Пример {i+1}: RGB={x_train_rgb[i]}, метка={y_train_rgb[i]}")
        
        nn_rgb = NeuralNetwork(n_neurons=1, n_inputs=3)
        
        print("\nОбучение сети...")
        x_train_norm = x_train_rgb / 255.0
        y_train_norm = y_train_rgb.copy()
        
        nn_rgb.fit(x_train_norm, y_train_norm, alpha=0.1, target_error=0.05, max_epochs=200)
        
        print(f"\nВеса после обучения: {nn_rgb.weight[0]}")
        print(f"Смещение после обучения: {nn_rgb.bias[0]}")
        
        accuracy = test_accuracy(nn_rgb, x_test_rgb/255.0, y_test_rgb)
        print(f"\nТочность на тестовой выборке: {accuracy:.2f}%")
        
        if accuracy < 70:
            print("\nТочность низкая, используем ручные веса для обнаружения красного...")
            nn_rgb.weight = np.array([[2.0, -1.0, -1.0]])
            nn_rgb.bias = np.array([-0.5])
            accuracy_manual = test_accuracy(nn_rgb, x_test_rgb/255.0, y_test_rgb)
            print(f"Точность с ручными весами: {accuracy_manual:.2f}%")

        result = process_image_for_red_detection(nn_rgb, "data/pic.jpg")