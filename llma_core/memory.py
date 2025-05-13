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
