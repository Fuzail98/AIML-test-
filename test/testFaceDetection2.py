import cv2
import mediapipe as mp
import csv
import threading


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def testFaceDetection2(camera):
    cap = cv2.VideoCapture()
    print(f"Connecting to camera: {camera['Name']}")
    cap.open(f"rtsp://{camera['username']}:{camera['password']}@{camera['ipAddr']}:{camera['port']}")

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
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
            
            cv2.imshow(camera['Name'], image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cap.release()




with open('cameraList.csv', 'r') as f:
    cameraList = csv.reader(f, delimiter=':', lineterminator='\n')
    cameraDict = {}
    for cameras in cameraList:
        camera = {
            'ipAddr': cameras[1], 
            'port': cameras[2],
            'Name': cameras[3], 
            'username': cameras[4], 
            'password': cameras[5]
        }
        tmp = {}
        for serialNumber in [cameras[0]]:
            tmp["camera_%s" % serialNumber] = camera
            cameraDict.update(tmp)


threads = list()
for cameraNumber, cameraDetails in cameraDict.items():
    th = threading.Thread(target=testFaceDetection2, args=(cameraDetails,))
    threads.append(th)

for th in threads:
    th.start()

for th in threads:
    th.join()