import cv2


cap = cv2.VideoCapture()
cap.open("rtsp://admin:Shazabadmin123@172.16.1.6:554")


while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break