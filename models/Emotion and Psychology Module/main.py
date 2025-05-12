class EmotionModule:
    def __init__(self):
        self.emotion_state = "neutral"

    def update_emotion(self, stimulus):
        # Определяем эмоцию по внешнему стимулу (например, текст или аудио)
        if stimulus == "positive":
            self.emotion_state = "happy"
        elif stimulus == "negative":
            self.emotion_state = "sad"
        else:
            self.emotion_state = "neutral"

    def get_current_emotion(self):
        return self.emotion_state
