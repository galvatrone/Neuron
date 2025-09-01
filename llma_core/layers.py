"""
layers.py — нейронные слои и attention
--------------------------------------
Содержит строительные блоки модели:
- Dense слой // 
- LayerNorm
- MultiHeadAttention
- FeedForward нейросеть
"""
# layers.py

import numpy as np
# layers.py

class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Инициализация матрицы эмбеддингов
        self.weights = np.random.randn(vocab_size, embedding_dim)

    def forward(self, x):
        # x должен быть матрицей размера (batch_size, seq_length)
        if len(x.shape) == 1:  # Если x одномерный, добавляем batch размерность
            x = x[np.newaxis, :]
        return self.weights[x]  # Преобразуем токены в эмбеддинги

class Dense:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.zeros(output_dim)

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones((1, 1, dim))  # Размерность соответствует embedding_dim
        self.beta = np.zeros((1, 1, dim))  # Размерность соответствует embedding_dim
        self.eps = eps

    def forward(self, x):
        # x имеет форму (batch_size, seq_length, embedding_dim)
        mean = np.mean(x, axis=-1, keepdims=True)  # Среднее по embedding_dim
        var = np.var(x, axis=-1, keepdims=True)    # Дисперсия по embedding_dim
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # для числовой стабильности
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class MultiHeadAttention:
    def __init__(self, num_heads, embedding_dim):
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim  # Размерность эмбеддинга

        self.Wq = np.random.randn(embedding_dim, embedding_dim)
        self.Wk = np.random.randn(embedding_dim, embedding_dim)
        self.Wv = np.random.randn(embedding_dim, embedding_dim)

    def forward(self, x):
        # Проверяем форму входа
        if len(x.shape) != 3:
            raise ValueError(f"Expected input of shape (batch_size, seq_length, embedding_dim), but got {x.shape}")

        Q = np.dot(x, self.Wq)
        K = np.dot(x, self.Wk)
        V = np.dot(x, self.Wv)

        batch_size, seq_length, _ = Q.shape  # Получаем размерности

        # Убедитесь, что размерности правильные для многоголового внимания
        if Q.shape[2] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, but got {Q.shape[2]}")

        # Перемешиваем и разделяем на головы
        Q = Q.reshape(batch_size, self.num_heads, seq_length, self.embedding_dim // self.num_heads)
        K = K.reshape(batch_size, self.num_heads, seq_length, self.embedding_dim // self.num_heads)
        V = V.reshape(batch_size, self.num_heads, seq_length, self.embedding_dim // self.num_heads)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.embedding_dim // self.num_heads)
        attention_weights = softmax(scores)
        output = np.matmul(attention_weights, V)

        output = output.reshape(batch_size, seq_length, self.embedding_dim)
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

