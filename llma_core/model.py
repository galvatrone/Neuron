from layers import Dense
import numpy as np

class MiniGPT:
    def __init__(self, vocab_size, embed_dim):
        self.embed = Dense(vocab_size, embed_dim)
        self.out = Dense(embed_dim, vocab_size)

    def __call__(self, x):
        h = self.embed(x)
        return self.out(h)
