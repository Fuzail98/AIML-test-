import cv2


cap1 = cv2.VideoCapture()
cap1.open("rtsp://admin:Shazabadmin123@172.16.1.8:554")
cap2 = cv2.VideoCapture()
cap2.open("rtsp://admin:Shazabadmin123@172.16.1.7:554")
cap3 = cv2.VideoCapture()
cap3.open("rtsp://admin:Shazabadmin123@172.16.1.6:554")

while(True):
    ret, frame = cap1.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

while(True):
    ret, frame = cap2.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

while(True):
    ret, frame = cap3.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break