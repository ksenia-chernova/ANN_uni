import numpy as np

class NeuralNetwork:
    weight: np.ndarray
    bias: np.ndarray
    
    def __init__(self, n_neurons: int, n_inputs: int):
        self.weight = np.random.uniform(0.001, 0.2, (n_neurons, n_inputs))
        self.bias = np.zeros(n_neurons)
    
    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        weighted_sum = np.dot(self.weight, x) + self.bias.reshape(-1, 1)
        return self._activation_function(weighted_sum).flatten()
    
    def fit_1(self, x: np.ndarray, y: np.ndarray, epsilon: float = 1e-6):
        for k in range(len(x)):
            x_k = x[k]
            y_k = y[k]
            y_pred = self.predict(x_k)
            error = y_k - y_pred
            
            for i in range(len(self.weight)):
                sum_weights = np.sum(self.weight[i])
                if abs(sum_weights) > epsilon:
                    delta = error[i] / sum_weights
                    self.weight[i] += delta
    
    def fit_2(self, x: np.ndarray, y: np.ndarray, alpha: float):
        for k in range(len(x)):
            x_k = x[k].reshape(-1, 1)
            y_k = y[k].reshape(-1, 1)
            y_pred = self.predict(x_k).reshape(-1, 1)
            error = y_pred - y_k
            delta = alpha * np.dot(error, x_k.T)
            self.weight -= delta
    
    def fit(self, x: np.ndarray, y: np.ndarray, alpha: float = 0.01, target_error: float = 0.001, max_epochs: int = 1000):
        epoch = 0
        best_weight = self.weight.copy()
        best_bias = self.bias.copy()
        best_error = float('inf')
        
        while epoch < max_epochs:
            total_error = 0
            weight_copy = self.weight.copy()
            
            for k in range(len(x)):
                x_k = x[k].reshape(-1, 1)
                y_k = y[k].reshape(-1, 1)
                y_pred = self.predict(x_k).reshape(-1, 1)
                error = y_pred - y_k
                delta = alpha * np.dot(error, x_k.T)
                self.weight -= delta
                
                if np.any(np.isnan(self.weight)) or np.any(np.isinf(self.weight)):
                    self.weight = weight_copy
                    break
                    
                total_error += np.sum(0.5 * (error ** 2))
            
            if np.any(np.isnan(self.weight)) or np.any(np.isinf(self.weight)):
                self.weight = best_weight
                self.bias = best_bias
                break
            
            mse = total_error / len(x)
            
            if mse < best_error:
                best_error = mse
                best_weight = self.weight.copy()
                best_bias = self.bias.copy()
            
            if mse < target_error:
                break
                
            epoch += 1
        
        self.weight = best_weight
        self.bias = best_bias
        
        if epoch == max_epochs:
            print(f"Достигнуто максимальное количество эпох: {max_epochs}, Лучшая MSE: {best_error:.6f}")
        else:
            print(f"Обучение завершено на эпохе {epoch+1}, MSE={best_error:.6f}")