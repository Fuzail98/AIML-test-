import cv2

# Open the first camera stream
cap1 = cv2.VideoCapture()
cap1.open("rtsp://admin:Shazabadmin123@172.16.1.5:554")

# Open the second camera stream
cap2 = cv2.VideoCapture()
# Specify the URL or video file path for the second camera
cap2.open("rtsp://admin:Shazabadmin123@172.16.1.6:554")

while True:
    # Read frames from the first camera
    ret1, frame1 = cap1.read()
    if not ret1:
        break

    # Read frames from the second camera
    ret2, frame2 = cap2.read()
    if not ret2:
        break

    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Resize frames to desired dimensions
    frame1 = cv2.resize(frame1, (512, 288))
    frame2 = cv2.resize(frame2, (512, 288))

    # Combine frames side by side
    combined_frame = cv2.hconcat([frame1, frame2])

    # Display the combined frame
    cv2.imshow('frame', combined_frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the capture objects and close the windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()