import numpy as np

# Функция активации (например, ReLU)
def relu(x):
    return np.maximum(0, x)

# Функция для вычисления выхода нейрона
def neuron(inputs, weights, bias):
    return relu(np.dot(inputs, weights) + bias)

# Пример использования
inputs = np.array([1.0, 2.0, 3.0])  # Входные данные
weights = np.array([0.5, 0.3, -0.7])  # Веса
bias = 2.0  # Смещение

output = neuron(inputs, weights, bias)
print("Выход нейрона:", output)

def self_attention(query, key, value):
    # Вычисление внимания (скорость внимания) для каждого слова
    attention_score = np.dot(query, key.T)  # Простейшая матрица внимания
    attention_weight = np.softmax(attention_score, axis=-1)
    output = np.dot(attention_weight, value)
    return output



class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Инициализация весов и смещений для всех слоев
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.random.rand(output_size)

    # Функция активации - ReLU для скрытых слоев, Sigmoid для выходного слоя
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    # Прогнозирование (forward pass)
    def predict(self, X):
        # Вход -> скрытый слой
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.relu(hidden_input)

        # Скрытый слой -> выходной слой
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output = self.sigmoid(output_input)

        return output

# Тестирование многослойной нейронной сети
input_size = 3  # 3 входа
hidden_size = 4  # 4 нейрона в скрытом слое
output_size = 10     # Один выход

# Создаем нейронную сеть
network = NeuralNetwork(input_size, hidden_size, output_size)

# Пример входных данных
input_data = np.array([0.1, 0.5, 0.9])

# Прогнозируем результат
output = network.predict(input_data)
print(f"Вход: {input_data}, Выход: {output}")
