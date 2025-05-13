"""
model.py — Архитектура нейросети (мини-GPT)
-------------------------------------------
Собирает слои из layers.py в единый трансформер-блок и модель.

- TransformerBlock: слой из attention + FFN + нормализация
- MiniGPT: простая LLM модель на основе нескольких блоков
"""

from layers import Dense, LayerNorm, MultiHeadAttention, FeedForward

class TransformerBlock:
    """
    Один трансформерный блок, который включает внимание и FeedForward.
    """
    def __init__(self, dim, num_heads):
        self.attention = MultiHeadAttention(num_heads, dim)
        self.feed_forward = FeedForward(dim)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

    def forward(self, x):
        """
        Прямой проход через трансформерный блок.
        """
        x = self.norm1.forward(x)
        attention_output = self.attention.forward(x)
        x = x + attention_output  # residual connection
        x = self.norm2.forward(x)
        ff_output = self.feed_forward.forward(x)
        return x + ff_output  # residual connection


class MiniGPT:
    """
    Мини-версии GPT с несколькими слоями.
    Включает несколько трансформерных блоков.
    """
    def __init__(self, num_layers, dim, num_heads):
        self.layers = [TransformerBlock(dim, num_heads) for _ in range(num_layers)]
        self.final_layer = Dense(dim, dim)  # Финальный слой для предсказания

    def forward(self, x):
        """
        Прямой проход через всю модель.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return self.final_layer.forward(x)
