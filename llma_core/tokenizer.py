# tokenizer.py

import numpy as np
import json
from collections import Counter


class Tokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}

    def fit(self, text_data):
        """
        Создаем словарь токенов на основе уникальных слов в тексте.
        Сохраняем слова в порядке их появления.
        """
        # Разбиваем текст на слова и собираем уникальные
        tokens = ' '.join(text_data).split()
        unique_tokens = sorted(set(tokens), key=tokens.index)  # Сохраняем порядок появления

        for idx, token in enumerate(unique_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def encode(self, text):
        """
        Кодируем текст в индексы, используя словарь токенов.
        """
        return [self.token_to_id[token] for token in text.split() if token in self.token_to_id]
    
    def decode(self, token_ids):
        """
        Декодируем индексы обратно в текст, используя словарь.
        """
        return ' '.join([self.id_to_token.get(idx, '<UNK>') for idx in token_ids])

class BPETokenizer:
    """
    Реализация BPE (Byte-Pair Encoding) токенизатора.
    BPE эффективно обрабатывает редкие слова, выделяя часто встречающиеся части.
    """
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merge_rules = {}

    def fit(self, texts):
        """
        Строим словарь пар байтов из текста.
        """
        # Разбиение текста на символы
        vocab = {}
        for text in texts:
            text = ' '.join(text.split())  # Преобразуем текст в формат с пробелами между буквами
            chars = list(text)  # Получаем список символов
            for char in chars:
                vocab[char] = vocab.get(char, 0) + 1

        # Создание начального словаря для BPE
        while len(vocab) < self.vocab_size:
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merge_vocab(best_pair, vocab)
            self.merge_rules[best_pair] = len(self.merge_rules)  # Сохраняем правило с номером шага

        self.vocab = vocab

    def get_stats(self, vocab):
        """Вычисляет частоты пар символов."""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        """Объединяет пару символов в один токен."""
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in list(vocab.keys()):
            new_word = word.replace(bigram, replacement)
            vocab[new_word] = vocab.pop(word)

    def encode(self, text):
        """
        Преобразует текст в последовательность токенов, используя BPE.
        """
        text = ' '.join(text.split())  # Разделяем текст на символы
        for pair in sorted(self.merge_rules, key=lambda x: self.merge_rules[x], reverse=True):
            text = text.replace(' '.join(pair), ''.join(pair))
        return text.split()  # Возвращаем токены

    def decode(self, tokens):
        """
        Декодирует токены обратно в текст.
        """
        return " ".join(tokens)

# Пример использования
if __name__ == "__main__":
    # Пример текста
    text_data = ["Пример текста для обучения токенизатора BPE."]

    # Инициализация и обучение токенизатора
    tokenizer = Tokenizer(vocab_size=100)
    tokenizer.fit(text_data)

    # Кодирование текста
    encoded_text = tokenizer.encode("Пример текста для обучения токенизатора BPE.")
    print(f"Закодированный текст: {encoded_text}")

    # Декодирование текста
    decoded_text = tokenizer.decode(encoded_text)
    print(f"Декодированный текст: {decoded_text}")