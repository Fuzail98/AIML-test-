import cv2

# Load pre-trained Haar cascade classifier
classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Define video capture object
cap = cv2.VideoCapture(0)

# Initialize counters and dictionary for IDs
enter_count = 0
exit_count = 0
people = {}

# Define initial position of door
door_pos = 0  # left: 0, right: 1

while True:
    # Read frame from video capture
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in frame using Haar cascade classifier
    faces = classifier.detectMultiScale(gray, 1.3, 5)

    # Draw bounding boxes around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Check if face is entering or leaving through door
        if x < door_pos and door_pos == 1:
            # Get ID of person based on position
            person_id = str(x) + '-' + str(y)
            # If person is entering for the first time, add to dictionary
            if person_id not in people:
                people[person_id] = 1
                enter_count += 1
            # Otherwise, increment the number of times the person has entered
            else:
                people[person_id] += 1
        elif x > door_pos and door_pos == 0:
            # Get ID of person based on position
            person_id = str(x) + '-' + str(y)
            # If person is in the dictionary, increment exit count and remove from dictionary
            if person_id in people:
                exit_count += 1
                del people[person_id]

    # Display frame with bounding boxes
    cv2.imshow('frame', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Print counts and dictionary
print('People entering:', enter_count)
print('People exiting:', exit_count)
print('People still inside:', len(people))
print('People:', people)
