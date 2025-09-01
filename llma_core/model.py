# simple_transformer.py
import numpy as np
from layers import Embedding, LayerNorm, MultiHeadAttention, FeedForward

class EncoderBlock:
    def __init__(self, embedding_dim, num_heads):
        self.mha = MultiHeadAttention(num_heads, embedding_dim)
        self.ff = FeedForward(embedding_dim)
        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)

    def forward(self, x):
        # Многоголовое внимание + остаток
        attn_out = self.mha.forward(x)
        x = self.norm1.forward(x + attn_out)

        # FeedForward + остаток
        ff_out = self.ff.forward(x)
        x = self.norm2.forward(x + ff_out)
        return x

class SimpleTransformer:
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, seq_length):
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.encoder_layers = [EncoderBlock(embedding_dim, num_heads) for _ in range(num_layers)]
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim

    def forward(self, x):
        x = self.embedding.forward(x)  # Токены в эмбеддинги
        for layer in self.encoder_layers:
            x = layer.forward(x)
        return x

# -------------------------------
# Пример использования
# -------------------------------
if __name__ == "__main__":
    vocab_size = 50
    embedding_dim = 64
    num_heads = 4
    num_layers = 32
    seq_length = 10
    batch_size = 30

    # Генерация случайных токенов
    x = np.random.randint(0, vocab_size, size=(batch_size, seq_length))

    model = SimpleTransformer(vocab_size, embedding_dim, num_heads, num_layers, seq_length)
    out = model.forward(x)
    
    print("Входная форма:", x.shape)
    print("Выходная форма:", out.shape)
    print("Выходной тензор:\n", out)
