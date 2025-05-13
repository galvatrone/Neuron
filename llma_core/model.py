# model.py

from layers import Embedding, Dense, LayerNorm, MultiHeadAttention

class TransformerModel:
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Слой эмбеддинга
        self.embedding = Embedding(vocab_size, embedding_dim)  
        
        # Многоголовые слои внимания
        self.attention_layers = [MultiHeadAttention(num_heads, embedding_dim) for _ in range(num_layers)]
        
        # Feed-forward слои
        self.feed_forward_layers = [Dense(embedding_dim, embedding_dim * 4) for _ in range(num_layers)]
        
        # Слой нормализации
        self.layer_norm_layers = [LayerNorm(embedding_dim) for _ in range(num_layers)]
        
        # Выходной слой
        self.output_layer = Dense(embedding_dim, vocab_size)

    def forward(self, x):
        # Преобразуем токены в эмбеддинги
        x = self.embedding.forward(x)
        
        # Применяем внимание и feed-forward слои с нормализацией
        for i in range(self.num_layers):
            x = self.attention_layers[i].forward(x)
            x = self.layer_norm_layers[i].forward(x)
            x = self.feed_forward_layers[i].forward(x)
            x = self.layer_norm_layers[i].forward(x)
        
        # Применяем выходной слой
        return self.output_layer.forward(x)
