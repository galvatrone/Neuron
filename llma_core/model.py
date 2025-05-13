# model.py

from layers import Dense, LayerNorm, MultiHeadAttention

class TransformerModel:
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.embedding = Dense(vocab_size, embedding_dim)  # Эмбеддинг для словаря
        self.attention_layers = [MultiHeadAttention(num_heads, embedding_dim) for _ in range(num_layers)]
        self.feed_forward_layers = [Dense(embedding_dim, embedding_dim * 4) for _ in range(num_layers)]
        self.output_layer = Dense(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding.forward(x)
        for i in range(self.num_layers):
            x = self.attention_layers[i].forward(x)
            x = self.feed_forward_layers[i].forward(x)
        return self.output_layer.forward(x)
