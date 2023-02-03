import cv2 as cv 
import datetime
import asyncio

async def addTimeStamp(frame):
    # Adding timestamp on the live stream
    font = cv.FONT_HERSHEY_COMPLEX
    dt = str(datetime.datetime.now())
    frame = cv.putText(frame, dt, (10, 100), font, 1, 
                    (210, 155, 155), 4, cv.LINE_8)
    return frame
