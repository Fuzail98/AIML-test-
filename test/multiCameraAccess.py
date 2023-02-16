import cv2 as cv
import threading
import csv
import os
import time


def runFaceDetection(frame):
    haar_cascades_path = os.path.dirname(cv.__file__) + "/data/haarcascade_frontalface_default.xml"
    
    face_cascade = cv.CascadeClassifier(haar_cascades_path)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    time.sleep(0.5)
    return frame


def read_frame(capture):
        capture.grab()
        ret, frame = capture.retrieve()
        if not ret:
            print('Empty Frame')
            return
        return frame


def multiCameraAccess(camera):
    cap = cv.VideoCapture()
    print(f"Connecting to camera: {camera['Name']}")
    cap.open(f"rtsp://{camera['username']}:{camera['password']}@{camera['ipAddr']}:{camera['port']}")

    while(True):
        frame = read_frame(cap)
        # task1 = asyncio.create_task(read_frame(frame)) 
        # task2 = asyncio.create_task(runFaceDetection(frame))
        # await asyncio.gather(task2)
        # await asyncio.sleep(0.01)
        frame = runFaceDetection(frame)
        cv.imshow(camera['ipAddr'], frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        print('####################')

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
    th = threading.Thread(target=multiCameraAccess, args=(cameraDetails,))
    threads.append(th)

for th in threads:
    th.start()

for th in threads:
    th.join()