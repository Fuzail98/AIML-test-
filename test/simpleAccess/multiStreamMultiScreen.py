import cv2
import json

# Load camera links from the JSON file
with open('../allCameras.json') as json_file:
    camera_links = json.load(json_file)

# Create a dictionary to store video captures for each camera
video_captures = {}

# Create OpenCV windows for displaying the streams
cv2.namedWindow('Camera Streams', cv2.WINDOW_NORMAL)

# Set the initial position for the first window
window_x, window_y = 0, 0

# Iterate through the camera links
for camera_name, camera_link in camera_links.items():
    # Create a video capture for the current camera link
    cap = cv2.VideoCapture(camera_link)
    
    # Check if the video capture was successfully opened
    if not cap.isOpened():
        print(f"Failed to open camera: {camera_name}")
        continue
    
    # Store the video capture in the dictionary
    video_captures[camera_name] = cap

    # Create a window for the current camera
    cv2.namedWindow(camera_name, cv2.WINDOW_NORMAL)

    # Set the position for the current window
    cv2.moveWindow(camera_name, window_x, window_y)

    # Update the window position for the next camera
    window_x += 640
    if window_x + 640 > 1920:
        window_x = 0
        window_y += 480

while True:
    # Iterate through the video captures
    for camera_name, cap in video_captures.items():
        # Read a frame from the current camera
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame from camera: {camera_name}")
            continue

        # Resize the frame to fit the display
        frame = cv2.resize(frame, (640, 480))

        # Display the frame in the corresponding window
        cv2.imshow(camera_name, frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video captures and close the windows
for cap in video_captures.values():
    cap.release()
cv2.destroyAllWindows()
