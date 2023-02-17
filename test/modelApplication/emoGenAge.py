import cv2
import json
import threading
import mediapipe as mp 
import time
import sys

from models import ageDetection, emotionDetection, genderDetection, fpsCounter
sys.path.append('E:/POCs/AIML/AIML-test-/test/mediapipeFaceDetection')
from mediapipeFaceDetection import mediapipeFaceDetecion

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def modelDetect(cameraName, cameraDetails):
    cap = cv2.VideoCapture()
    print(f"Connecting to camera: {cameraName}")
    cap.open(cameraDetails)

    previousTime = 0
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            currentTime = time.time()
            fps = 1/(currentTime - previousTime)
            previousTime = currentTime
            fpsCounter(image, fps)
            print(f'fps for  {cameraName} = {int(fps)}')

            # emotionDetection(image)
            ageDetection(image)
            # genderDetection(image)
            
            image = mediapipeFaceDetecion.faceDetection(image, mp_drawing, face_detection)
            
            cv2.imshow(cameraName, image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cap.release()

cameras = json.loads(open('../cameras.json').read())

threads = list()
for cameraName, cameraDetails in cameras.items():
    th = threading.Thread(target=modelDetect, args=(cameraName, cameraDetails))
    threads.append(th)

for th in threads:
    th.start()

for th in threads:
    th.join()