import cv2
import mediapipe as mp
import threading
import json
import time


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
            cv2.putText(image, f'fps = {int(fps)}', (20, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 2)
            print(f'fps = {int(fps)}: cam')
            if not success:
                print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
                continue
# To improve performance, optionally mark the image as not writeable to
    # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
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