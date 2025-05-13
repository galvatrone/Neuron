# train.py

import numpy as np
from model import TransformerModel
from tokenizer import Tokenizer
import config

def train(model, data, epochs, learning_rate):
    for epoch in range(epochs):
        for batch in data:
            predictions = model.forward(batch)
            loss = compute_loss(predictions, batch)
            gradients = compute_gradients(loss, model)
            update_weights(model, gradients, learning_rate)
        print(f'Epoch {epoch+1}/{epochs} - Loss: {loss}')

def compute_loss(predictions, targets):
    # Примерный расчёт потерь (кросс-энтропия)
    return np.mean((predictions - targets)**2)

def compute_gradients(loss, model):
    # Примерная функция для вычисления градиентов
    return np.ones_like(model.output_layer.weights) * 0.1

def update_weights(model, gradients, learning_rate):
    model.output_layer.weights -= learning_rate * gradients
