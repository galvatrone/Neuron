class CentralControlUnit:
    def __init__(self):
        self.perception = PerceptionModule()
        self.memory = MemoryModule()
        self.emotion = EmotionModule()
        self.logic = LogicModule(self.memory)

    def process_input(self):
        # Считывание данных и обработка
        self.perception.capture_image()
        self.perception.capture_audio()
        
        image, audio = self.perception.process_input()
        
        # Сохранение данных в память
        self.memory.store_experience({"image": image, "audio": audio})
        
        # Определение эмоций
        self.emotion.update_emotion("positive")
        
        # Принятие решения
        decision = self.logic.make_decision({"image": image, "audio": audio})
        
        return decision
