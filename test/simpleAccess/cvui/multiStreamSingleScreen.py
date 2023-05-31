import numpy as np
import cv2
import cvui
import json

WINDOW_NAME = 'CVUI Test'
GRID_SIZE = (1, 2)
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# Initialize cvui and create/open an OpenCV window.
cvui.init(WINDOW_NAME)

with open('../../allCameras.json') as f:
    cameras = json.load(f)

# Create a list to store the video capture objects
captures = []

# Open video capture for each camera
for camera_name, camera_url in cameras.items():
    print(f'Start capture of {camera_name}')
    cap = cv2.VideoCapture()
    print(f'End capture of {camera_name}...start open')
    cap.open(camera_url)
    print('create capture list')
    captures.append(cap)

# Create a blank canvas for the grid layout
print('Creating Empty canvas')
canvas = np.zeros((FRAME_HEIGHT * GRID_SIZE[0], FRAME_WIDTH * GRID_SIZE[1], 3), np.uint8)

while True:
    # Iterate over the video captures and display each camera feed in the grid
    for i, cap in enumerate(captures):
        print(f"Reading frames: {i + 1}")
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame from the video capture: {i}")
            continue

        # Resize each frame to fit the grid cells
        print('Resizing the frame')
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Calculate the row and column index for the current camera
        row = i // GRID_SIZE[1]
        col = i % GRID_SIZE[1]

        # Calculate the coordinates for the current grid cell
        x = col * FRAME_WIDTH
        y = row * FRAME_HEIGHT

        # Copy the frame onto the canvas at the corresponding grid cell
        print('Copying the frame on the canvas')
        canvas[y:y + FRAME_HEIGHT, x:x + FRAME_WIDTH] = frame

    # Show the canvas with the camera feeds using cvui
    print('Displaying Streams')
    cvui.imshow(WINDOW_NAME, canvas)
    print('########################################################################')
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release video captures and destroy OpenCV window
for cap in captures:
    cap.release()

cv2.destroyAllWindows()
