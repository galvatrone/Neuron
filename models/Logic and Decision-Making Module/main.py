class LogicModule:
    def __init__(self, memory_module):
        self.memory = memory_module
    
    def make_decision(self, current_data):
        # Логика принятия решения на основе текущих данных и памяти
        if "threat" in current_data:
            return "escape"
        elif "reward" in current_data:
            return "approach"
        else:
            return "wait"
