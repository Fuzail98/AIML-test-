import cv2
import cvui
import json

width = 1280
height = 720

# print(dir(cvui))

# ['Block', 'CLICK', 'COLUMN', 'CVUI_ANTIALISED', 'CVUI_FILLED', 'Context', 'DOWN', 'IS_DOWN', 'Internal', 'LEFT_BUTTON', 
# 'Label', 'MIDDLE_BUTTON', 'Mouse', 'MouseButton', 'OUT', 'OVER', 'Point', 'RIGHT_BUTTON', 'ROW', 'Rect', 'Render', 
# 'Size', 'TRACKBAR_DISCRETE', 'TRACKBAR_HIDE_LABELS', 'TRACKBAR_HIDE_MIN_MAX_LABELS', 'TRACKBAR_HIDE_SEGMENT_LABELS', 
# 'TRACKBAR_HIDE_STEP_SCALE', 'TRACKBAR_HIDE_VALUE_LABEL', 'TrackbarParams', 'UP', 'VERSION', '__builtins__', 
# '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'beginColumn', 
# 'beginRow', 'button', 'checkbox', 'context', 'counter', 'cv2', 'cvui', 'endColumn', 'endRow', 'iarea', 'image', 
# 'imshow', 'init', 'lastKeyPressed', 'main', 'mouse', 'np', 'printf', 'rect', 'space', 'sparkline', 'sys', 'text', 
# 'trackbar', 'update', 'watch', 'window']


def createCamera(cam_no, cameraInfo):
    print(f'Setting up camera: {cam_no} : {cameraInfo["room"]}')
    if cameraInfo['rtsp'] == True and cameraInfo['cameraIndex'] == cam_no:
        cameraInfo["rtspLink"] = f"rtsp://{cameraInfo['username']}:{cameraInfo['password']}@{cameraInfo['ipAddress']}:554"
        rtsp = cameraInfo["rtspLink"]
        cap = cv2.VideoCapture()
        cap.open(rtsp)
        return cap
    else:
        print('No rtsp link found OR camera does not exist in configuration.')
        return 0

with open("../cameras.json") as f:
    cameras = json.load(f)

cam_no = list(cameras.values())[0]['cameraIndex'] # 1

initialCamera = list(cameras.values())[0]

cam = createCamera(cam_no, initialCamera)
cvui.init('screen')

while True:
    ret, frame = cam.read()
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if (cvui.button(frame, width - 100, height - 40, "Next") and cvui.mouse(cvui.CLICK)):
        print("Next Button Pressed")
        cvui.init('screen')
        cam_no = cam_no + 1
        if (cam_no > len(cameras)):
            cam_no = 1
        del cam
        cameraInfo = list(cameras.values())[cam_no - 1]
        cam = createCamera(cam_no, cameraInfo)
        
    if (cvui.button(frame, width - 200, height - 40, "Previous") and cvui.mouse(cvui.CLICK)):
        print("Previous Button Pressed")
        cvui.init('screen')
        cam_no = cam_no - 1
        if (cam_no < 1):
            cam_no = len(cameras)
        del cam
        cameraInfo = list(cameras.values())[cam_no - 1]
        cam = createCamera(cam_no, cameraInfo)

    if (cvui.button(frame, width - 270, height - 40, "Exit") and cvui.mouse(cvui.CLICK)):
        print('Exiting...')
        cv2.destroyAllWindows()
        break
        
    cv2.imshow('screen', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
