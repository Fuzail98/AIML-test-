import cv2 as cv
# from test import read_frame
from video_async import MultiCameraCapture
import json
from add_datetime import addTimeStamp
from faceDetection import runFaceDetection


if __name__ == "__main__":
    cameras = json.loads(open('cameras.json').read())
    captured = MultiCameraCapture(sources=cameras)
    # cap = cv.VideoCapture(0)
    # assert cap.isOpened()
    # print(cap)
    while True:
        for camera_name, cap in captured.captures.items():
            frame = captured.read_frame(cap)

            frame = addTimeStamp(frame)
            frame = runFaceDetection(frame)
            
            cv.imshow(camera_name, frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()