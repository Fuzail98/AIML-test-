import cv2
import time
import threading
import json


def multiCameraAccessJSON(cameraName, cameraDetails):
    cap = cv2.VideoCapture()
    print(f"Connecting to camera: {cameraName}")
    cap.open(cameraDetails)
    previousTime = 0
    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            print('Ignoring camera Frame')
            continue

        currentTime = time.time()
        fps = 1/(currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(frame, f'fps = {int(fps)}', (20, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 2)


        cv2.imshow(cameraName, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()




cameras = json.loads(open('../allCameras.json').read())

threads = list()
for cameraName, cameraDetails in cameras.items():
    th = threading.Thread(target=multiCameraAccessJSON, args=(cameraName, cameraDetails))
    threads.append(th)

for th in threads:
    th.start()

for th in threads:
    th.join()
