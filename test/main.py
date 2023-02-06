import cv2 as cv
# from test import read_frame
from video_async import MultiCameraCapture
import json
from add_datetime import addTimeStamp
from faceDetection import runFaceDetection
import asyncio


async def run_fd_time(frame):
    task1 = asyncio.create_task(addTimeStamp(frame)) 
    task2 = asyncio.create_task(runFaceDetection(frame))
    await asyncio.gather(task1, task2)
    await asyncio.sleep(0.01)


async def main(captured_obj):
    while True:
        async for camera_name, cap in captured.asyncCameraGen():
            frame = await captured.read_frame(cap)
            # frame = await addTimeStamp(frame)
            # frame = runFaceDetection(frame)

            await run_fd_time(frame)
            await captured_obj.showFrame(camera_name, frame)
            # cv.imshow(camera_name, frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    cameras = json.loads(open('cameras.json').read())
    captured = MultiCameraCapture(sources=cameras)

    asyncio.run(main(captured_obj=captured))