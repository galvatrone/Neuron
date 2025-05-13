from layers import Embedding, Dense, LayerNorm, MultiHeadAttention

class TransformerModel:
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, dim):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim = dim  # Новый аргумент для размерности выходного пространства

        # Эмбеддинг для словаря
        self.embedding = Embedding(vocab_size, embedding_dim)

        # Инициализация слоев многоголового внимания, теперь передаем 'dim'
        self.attention_layers = [MultiHeadAttention(num_heads, embedding_dim, dim) for _ in range(num_layers)]

        # Инициализация слоев для пропускания через слой FeedForward
        self.feed_forward_layers = [Dense(embedding_dim, embedding_dim * 4) for _ in range(num_layers)]

        # Выходной слой
        self.output_layer = Dense(embedding_dim, vocab_size)

    def forward(self, x):
        # Прямой проход через эмбеддинг
        x = self.embedding.forward(x)

        # Проход через все слои внимания и feed-forward
        for i in range(self.num_layers):
            x = self.attention_layers[i].forward(x)
            x = self.feed_forward_layers[i].forward(x)
        
        # Выходной слой
        return self.output_layer.forward(x)
