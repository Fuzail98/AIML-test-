import cv2
import asyncio
import numpy as np

class MultiCameraCapture:
    def __init__(self, sources: dict) -> None:
        assert sources
        print(sources)

        self.captures = {}
        for camera_name, link in sources.items():
            cap = cv2.VideoCapture(link)
            print(camera_name)
            assert cap.isOpened()
            self.captures[camera_name] = cap 

    @staticmethod
    def read_frame(capture):
        capture.grab()
        ret, frame = capture.retrieve()
        if not ret:
            print('Empty Frame')
            return
        return frame

    @staticmethod
    async def showFrame(windowName: str, frame: np.array):
        cv2.imshow(windowName, frame)

    async def asyncCameraGen(self):
        for camera_name, capture in self.captures.items():
            yield camera_name, capture
            await asyncio.sleep(0.001)