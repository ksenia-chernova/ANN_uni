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
    
    def fit_1(self, x: np.ndarray, y: np.ndarray):
        for k in range(len(x)):
            x_k = x[k]
            y_k = y[k]
            y_pred = self.predict(x_k)
            
            for i in range(len(self.weight)):
                sum_x = np.sum(x_k)
                if abs(sum_x) > 1e-10:
                    delta = (y_k[i] - y_pred[i]) / sum_x
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
        
        while epoch < max_epochs:
            total_error = 0
            
            for k in range(len(x)):
                x_k = x[k].reshape(-1, 1)
                y_k = y[k].reshape(-1, 1)
                y_pred = self.predict(x_k).reshape(-1, 1)
                error = y_pred - y_k
                delta = alpha * np.dot(error, x_k.T)
                self.weight -= delta
                
                total_error += np.sum(0.5 * (error ** 2))
            
            mse = total_error / len(x)
            
            if mse < target_error:
                break
                
            epoch += 1