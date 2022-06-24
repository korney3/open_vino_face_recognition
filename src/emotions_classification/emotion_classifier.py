import os

import numpy as np
from keras.layers import BatchNormalization
from keras_preprocessing.image import array_to_img
from tensorflow.keras.models import load_model
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.python.keras.models import model_from_json
import tensorflow as tf

from utils import cut_rois


class EmotionClassifier:
    def __init__(self):
        script_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_path, "../models_weights")
        self.model = load_model(os.path.join(model_path, 'model_emotions'))
        self.labels_dict =  {0:'Злость',1:'Отвращение',2:'Страх',3:'Радость',4:'Нейтральное состояние',5:'Грусть',6:'Удивление'}

    def process(self, frame, rois):
        inputs = self.image_preprocessing(frame, rois)
        results = [self.make_predictions(input) for input in inputs]
        return results


    def image_preprocessing(self, frame, rois):
        inputs = cut_rois(frame, rois)
        gb_inputs = list(map(lambda x: tf.image.rgb_to_grayscale(x), inputs))
        resize_inputs = list(map(lambda x: tf.image.resize(x, (48, 48)), gb_inputs))
        return resize_inputs

    def make_predictions(self, img):
        img = np.expand_dims(img, axis=0)  # makes image shape (1,48,48)
        img = img.reshape(1, 48, 48, 1)
        result = self.model.predict(img)
        result = list(result[0])
        result_naming = dict([(self.labels_dict[i], result[i]) for i in range(0,7)])
        return result_naming


emotion_classifier = EmotionClassifier()
print()