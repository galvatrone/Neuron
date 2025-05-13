"""
layers.py — нейронные слои и attention
--------------------------------------
Содержит строительные блоки модели:
- Dense слой
- LayerNorm
- MultiHeadAttention
- FeedForward нейросеть
"""

import numpy as np

class Dense:
    """
    Полносвязный слой.
    Каждое входное значение соединяется с каждым выходным значением.
    """
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim)  # Инициализация весов
        self.bias = np.zeros(output_dim)  # Инициализация смещений

    def forward(self, x):
        """
        Прямой проход через слой.
        """
        return np.dot(x, self.weights) + self.bias


class LayerNorm:
    """
    Нормализация данных на уровне слоя.
    Это помогает стабилизировать обучение.
    """
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)  # Масштабирующий коэффициент
        self.beta = np.zeros(dim)  # Смещение
        self.eps = eps  # Маленькое значение для стабильности

    def forward(self, x):
        """
        Нормализуем данные.
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta


class MultiHeadAttention:
    """
    Механизм многоголового внимания (multi-head attention).
    Используется для того, чтобы модель могла учитывать контекст в разных аспектах.
    """
    def __init__(self, num_heads, dim):
        self.num_heads = num_heads
        self.dim = dim

        # Инициализация весов для каждой головы внимания
        self.Wq = np.random.randn(dim, dim)
        self.Wk = np.random.randn(dim, dim)
        self.Wv = np.random.randn(dim, dim)

    def forward(self, x):
        """
        Прямой проход через слой внимания.
        """
        Q = np.dot(x, self.Wq)
        K = np.dot(x, self.Wk)
        V = np.dot(x, self.Wv)

        scores = np.dot(Q, K.T) / np.sqrt(self.dim)
        attention_weights = softmax(scores)  # softmax для получения вероятностей внимания

        output = np.dot(attention_weights, V)
        return output


class FeedForward:
    """
    Полносвязная нейросеть с активацией.
    Состоит из двух слоев с функцией активации между ними.
    """
    def __init__(self, dim):
        self.dense1 = Dense(dim, dim * 4)
        self.dense2 = Dense(dim * 4, dim)

    def forward(self, x):
        """
        Прямой проход через FeedForward слой.
        """
        x = self.dense1.forward(x)
        x = np.maximum(0, x)  # Применяем ReLU
        x = self.dense2.forward(x)
        return x
