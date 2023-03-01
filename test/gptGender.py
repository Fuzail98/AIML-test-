import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('age_detection_model.h5')

# Load the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a video frame
    ret, frame = cap.read()

    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (224, 224))
    gray = np.expand_dims(gray, axis=0)
    gray = np.expand_dims(gray, axis=-1)

    # Make predictions
    prediction = model.predict(gray)

    # Display the predictions on the video frame
    age = np.argmax(prediction)
    cv2.putText(frame, str(age), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Show the video frame
    cv2.imshow('Age Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
