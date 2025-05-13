# main.py

import numpy as np
from model import TransformerModel
from tokenizer import Tokenizer
from train import train
import config
import memory

# Загружаем параметры из config
vocab_size = config.CONFIG['vocab_size']
embedding_dim = config.CONFIG['embedding_dim']
num_heads = config.CONFIG['num_heads']
num_layers = config.CONFIG['num_layers']
learning_rate = config.CONFIG['learning_rate']
epochs = config.CONFIG['epochs']

# Инициализация токенизатора
tokenizer = Tokenizer(vocab_size)
tokenizer.fit(["Пример текста для обучения"])  # Подгонка токенизатора

# Создание модели
model = TransformerModel(vocab_size, embedding_dim, num_layers, num_heads)

# Инициализация данных для тренировки
data = np.random.randint(0, vocab_size, (100, config.CONFIG['sequence_length']))  # Пример данных

# Обучение модели
train(model, data, epochs, learning_rate)

# Сохранение модели
memory.save_model(model, "transformer_model.pkl")
memory.save_model_json(model, "transformer_model.json")

# Загружаем модель
loaded_model = memory.load_model("transformer_model.pkl")
