import os

from tensorflow.keras.models import load_model


class EmotionClassifier:
    def __init__(self):
        script_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_path, "../models_weights")
        self.model = load_model(os.path.join(model_path, 'kaggle_model_emotions.h5'))

emotion_classifier = EmotionClassifier()
print()