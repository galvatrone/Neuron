"""
layers.py — нейронные слои и attention
--------------------------------------
Содержит строительные блоки модели:
- Dense слой
- LayerNorm
- MultiHeadAttention
- FeedForward нейросеть
"""
# layers.py

import numpy as np

class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        # создаём веса размером (vocab_size, embedding_dim)
        self.weights = np.random.randn(vocab_size, embedding_dim)

    def forward(self, token_ids):
        # token_ids должен быть двумерным массивом (batch_size, seq_length)
        return self.weights[token_ids]  # извлекаем векторы по ID токенов

class Dense:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.zeros(output_dim)

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # для числовой стабильности
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class MultiHeadAttention:
    def __init__(self, num_heads, dim, embedding_dim):
        self.num_heads = num_heads
        self.dim = dim
        # Исправляем инициализацию весов
        self.Wq = np.random.randn(embedding_dim, dim)  # теперь размерность (512, dim)
        self.Wk = np.random.randn(embedding_dim, dim)  # аналогично для Wk
        self.Wv = np.random.randn(embedding_dim, dim)  # аналогично для Wv

    def forward(self, x):
        # Проверяем размерности для матричного умножения
        Q = np.dot(x, self.Wq)  # x: (128, 512), Wq: (512, dim)
        K = np.dot(x, self.Wk)  # K: (128, dim)
        V = np.dot(x, self.Wv)  # V: (128, dim)

        # Далее вычисляем внимание
        scores = np.dot(Q, K.T) / np.sqrt(self.dim)
        attention_weights = softmax(scores)
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

