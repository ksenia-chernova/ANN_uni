import numpy as np

class Neuron:
    def __init__(self, w: np.ndarray[float] = None, b: float = None):
        if w is not None:
            self.weight = w
        if b is not None:
            self.bias = b

    def __threshold_function(self, v: float) -> int:
        return 1 if v >= 0 else 0

    def predict(self, x: np.ndarray[int]) -> int:
        u = 0
        for i in range(len(x)):
            u += self.weight[i] * x[i]
        return self.__threshold_function(u + self.bias)


if __name__ == "__main__":
    weights = np.array([-1, -1])
    bias = 1

    neuron = Neuron(weights, bias)

    test_inputs = [
        np.array([0, 0]),
        np.array([0, 1]), 
        np.array([1, 0]),
        np.array([1, 1])
    ]
    
    for x in test_inputs:
        output = neuron.predict(x)
        print(f" {x[0]} |  {x[1]} |   {output}")
    
    expected_outputs = [1, 1, 1, 0]
    actual_outputs = [neuron.predict(x) for x in test_inputs]
    
    if actual_outputs == expected_outputs:
        print("Нейрон работает корректно")
    else:
        print("Нейрон работает некорректно")