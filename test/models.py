from deepface import DeepFace
import cv2
import time

font = cv2.FONT_HERSHEY_COMPLEX

def emotionDetection(image):
    emotion = DeepFace.analyze(image, actions=['emotion'])
    cv2.putText(
        image,
        f"Emo:{emotion[0]['dominant_emotion']}",
        (20, 200),
        font, 1,
        (0, 0, 255), 2
    )

def ageDetection(image):
    age = DeepFace.analyze(image, actions=['age'])
    age[0]['age'] = str(age[0]['age'])
    cv2.putText(
        image,
        f"Age:{age[0]['age']}",
        (20, 400),
        font, 3,
        (0, 0, 255), 2
    )

def genderDetection(image):
    gender = DeepFace.analyze(image, actions=['gender'])
    cv2.putText(
        image,
        f"Gen:{gender[0]['dominant_gender']}",
        (20, 300),
        font, 1,
        (0, 0, 255), 2
    )

def fpsCounter(image, fps):

    cv2.putText(
        image, 
        f'fps = {int(fps)}', 
        (20, 100), 
        font, 1, 
        (255, 0, 0), 2
    )