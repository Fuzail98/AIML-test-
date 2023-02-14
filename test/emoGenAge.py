import cv2
from deepface import DeepFace
import json
import threading
import mediapipe as mp 
import time

# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# cap = cv2.VideoCapture()
# cap.open(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Ignoring empty frame")
#         continue
#     result = DeepFace.analyze(frame, actions=['emotion'])
#     print(result[0])

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
#     faces = faceCascade.detectMultiScale(gray, 1.1, 4)

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

#     font = cv2.FONT_HERSHEY_SIMPLEX

    # cv2.putText(
    #     frame,
    #     result[0]['dominant_emotion'],
    #     (50, 50),
    #     font, 3,
    #     (0, 0, 255), 2
    # )

#     cv2.imshow("Frontal camera", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def testFaceDetection2(cameraName, cameraDetails):
    cap = cv2.VideoCapture()
    print(f"Connecting to camera: {cameraName}")
    cap.open(cameraDetails)

    previousTime = 0
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            currentTime = time.time()
            fps = 1/(currentTime - previousTime)
            previousTime = currentTime
            result = DeepFace.analyze(image, actions=['emotion', 'gender', 'age'])
            result[0]['age'] = str(result[0]['age'])
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                image, 
                f'fps = {int(fps)}', 
                (20, 100), 
                font, 1, 
                (255, 0, 0), 2
            )
            cv2.putText(
                image,
                f"Emo:{result[0]['dominant_emotion']}",
                (20, 200),
                font, 1,
                (0, 0, 255), 2
            )
            cv2.putText(
                image,
                f"Gen:{result[0]['dominant_gender']}",
                (20, 300),
                font, 1,
                (0, 0, 255), 2
            )
            cv2.putText(
                image,
                f"Age:{result[0]['age']}",
                (20, 400),
                font, 3,
                (0, 0, 255), 2
            )

            print(f'fps for  {cameraName} = {int(fps)}')
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
            
            cv2.imshow(cameraName, image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cap.release()

threads = list()
cameras = json.loads(open('cameras.json').read())

for cameraName, cameraDetails in cameras.items():
    th = threading.Thread(target=testFaceDetection2, args=(cameraName, cameraDetails))
    threads.append(th)

for th in threads:
    th.start()

for th in threads:
    th.join()