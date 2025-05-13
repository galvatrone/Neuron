"""
memory.py — временное хранилище истории диалога
-----------------------------------------------
Позволяет сохранять и читать контекст между запросами пользователя.
"""
class Memory:
    def __init__(self):
        self.history = []

    def add(self, message):
        """
        Добавление нового сообщения в память.
        """
        self.history.append(message)

    def get_context(self):
        """
        Получение последних 10 сообщений для контекста.
        """
        return self.history[-10:]
    
# memory.py

import pickle
import json


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def save_model_json(model, filename):
    with open(filename, 'w') as f:
        json.dump(model, f)

def load_model_json(filename):
    with open(filename, 'r') as f:
        model = json.load(f)
    return model

