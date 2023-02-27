
# import necessary packages
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os
import cvlib as cv
from keras.models import model_from_json
from keras.preprocessing import image


# We want it to run on CPU instead of GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# download pre-trained model file (one-time download)
dwnld_link = "https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model"
model_path = get_file("gender_detection.model", dwnld_link,
                     cache_subdir="pre-trained", cache_dir=os.getcwd())

model = load_model(model_path)

json_file = open('/home/shazab/AIML-test-/emogenage/Facial-emotion-recognition/top_models/fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model1 = model_from_json(loaded_model_json)
model1.load_weights('/home/shazab/AIML-test-/emogenage/Facial-emotion-recognition/top_models/fer.h5')

model_age = load_model('/home/shazab/AIML-test-/emogenage/Age-Gender-Prediction/model/model_age.hdf5')
# load model


# open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

classes = ['man','woman']

# loop through frames
while webcam.isOpened():

    # read frame from webcam
    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        exit()

    # apply face detection
    face, confidence = cv.detect_face(frame)

    print(face)
    print(confidence)

    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = gray_img[startY:endY,startX:endX]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255.0

        # apply gender detection on face
        conf = model.predict(face_crop)[0]
        predictions = model1.predict(img_pixels)
        max_index = int(np.argmax(predictions))

        emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
        predicted_emotion = emotions[max_index]
        img_detect = cv2.resize(frame[startY:endY,startX:endX], dsize=(50, 50)).reshape(1, 50, 50, 3)
        #age
        age = model_age.predict(img_detect/255.)[0][0]

        print(conf)
        print(classes)
        

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]
        
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)
       
        Y = startY - 10 if startY - 10 > 10 else startY + 10
       
        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (int(startX+20), int(Y-60)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.putText(frame, f'Age: {age}', (startX-25, Y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 255, 255), 2 )

    # display output
    cv2.imshow("gender and emotion detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()