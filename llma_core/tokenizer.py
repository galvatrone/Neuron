"""
tokenizer.py — модуль токенизации текста
----------------------------------------
Отвечает за преобразование текста в токены (числа) и обратно.

Содержит:
- SimpleTokenizer: простая токенизация по словам
- BPETokenizer: реализация BPE (Byte-Pair Encoding) токенизатора

Дальнейшее улучшение:
- Поддержка SentencePiece
- Сохранение и загрузка словаря
"""

class SimpleTokenizer:
    """
    Простая токенизация по словам.
    Разбивает текст на слова и создает словарь.
    """
    def __init__(self):
        self.vocab = {}  # Словарь для отображения слов в индексы и наоборот
        self.reverse_vocab = {}

    def fit(self, texts):
        """
        Обучаем токенизатор на тексте.
        Строим словарь из текста.
        """
        for text in texts:
            for word in text.split():
                if word not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[word] = idx
                    self.reverse_vocab[idx] = word

    def encode(self, text):
        """
        Преобразует текст в последовательность токенов.
        """
        return [self.vocab.get(word, self.vocab.get("<UNK>")) for word in text.split()]

    def decode(self, tokens):
        """
        Преобразует последовательность токенов обратно в текст.
        """
        return " ".join([self.reverse_vocab.get(token, "<UNK>") for token in tokens])


class BPETokenizer:
    """
    Реализация BPE (Byte-Pair Encoding) токенизатора.
    BPE эффективно обрабатывает редкие слова, выделяя часто встречающиеся части.
    """
    def __init__(self):
        self.vocab = {}

    def fit(self, texts):
        """
        Строим словарь пар байтов из текста.
        """
        for text in texts:
            # Кодирование текста в пары байтов
            pass  # Это упрощено для примера

    def encode(self, text):
        """
        Преобразует текст в последовательность токенов, используя BPE.
        """
        return []  # Здесь будет кодировка

    def decode(self, tokens):
        """
        Декодирует токены обратно в текст.
        """
        return " ".join(tokens)  # Здесь будет декодировка
