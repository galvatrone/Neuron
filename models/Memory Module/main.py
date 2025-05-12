class MemoryModule:
    def __init__(self):
        self.memory = []

    def store_experience(self, experience):
        self.memory.append(experience)

    def retrieve_memory(self):
        return self.memory
    
    def update_memory(self, new_data):
        # Обновляем или заменяем старые данные новыми
        self.memory.append(new_data)
