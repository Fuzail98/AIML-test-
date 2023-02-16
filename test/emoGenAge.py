import cv2
from deepface import DeepFace
import json
import threading
import mediapipe as mp 
import time

from putTextOnStream import putTextOnStream
from mediapipeFaceDetecion import faceDetection


def fpsCounter():
    previousTime = 0
    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime

    return fps

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def modelDetect(cameraName, cameraDetails):
    cap = cv2.VideoCapture()
    print(f"Connecting to camera: {cameraName}")
    cap.open(cameraDetails)

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            result = DeepFace.analyze(image, actions=['emotion', 'gender', 'age'])
            result[0]['age'] = str(result[0]['age'])

            fps = fpsCounter()

            putTextOnStream(image, fps, result)
            # print(f'fps for  {cameraName} = {int(fps)}')
            
            image = faceDetection(image, mp_drawing, face_detection)
            
            cv2.imshow(cameraName, image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cap.release()

threads = list()
cameras = json.loads(open('cameras.json').read())

for cameraName, cameraDetails in cameras.items():
    th = threading.Thread(target=modelDetect, args=(cameraName, cameraDetails))
    threads.append(th)

for th in threads:
    th.start()

for th in threads:
    th.join()