import cv2
from keras.models import load_model
import numpy as np

from utils.preprocessor import preprocess_input

# parameters for loading data and images
emotion_model_path = 'fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = {0:'angry',1:'disgust',2:'fear',3:'happy',
                4:'sad',5:'surprise',6:'neutral'}

emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

def get_emotion(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    try:
        gray_face = cv2.resize(gray, (emotion_target_size))
    except:
        return None

    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_prediction = emotion_classifier.predict(gray_face)
    emotion_probability = np.max(emotion_prediction)
    emotion_label_arg = np.argmax(emotion_prediction)
    emotion_text = emotion_labels[emotion_label_arg]

    return emotion_text, emotion_probability

