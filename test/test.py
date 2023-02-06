import os
import time
import json
import numpy as np
import cv2 as cv
import sys


def runFaceDetection(frame):
    haar_cascades_path = os.path.dirname(cv.__file__) + "/data/haarcascade_frontalface_default.xml"
    face_cascade = cv.CascadeClassifier(haar_cascades_path)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    time.sleep(0.5)
    return frame

class MultiCameraCapture:
    def __init__(self, sources: dict) -> None:
        assert sources
        print(sources)

        self.captures = {}
        for camera_name, link in sources.items():
            cap = cv.VideoCapture(link)
            print(camera_name)
            # assert cap.isOpened()
            self.captures[camera_name] = cap 

    @staticmethod
    def read_frame(capture):
        capture.grab()
        ret, frame = capture.retrieve()
        if not ret:
            print('Empty Frame')
            return
        return frame

def testFaceDetection1(frame):
    cascPath = sys.argv[1]
    faceCascade = cv.CascadeClassifier(cascPath)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30, 30),
        flags = cv.cv.CV_HAAR_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    time.sleep(0.5)
    return frame


if __name__ == "__main__":
    cameras = json.loads(open('cameras.json').read())
    captured = MultiCameraCapture(sources=cameras)

    while True:
        for camera_name, cap in captured.captures.items():
            frame = captured.read_frame(cap)
            # frame = runFaceDetection(frame)
            # frame = testFaceDetection1(frame)
            cv.imshow(camera_name, frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()