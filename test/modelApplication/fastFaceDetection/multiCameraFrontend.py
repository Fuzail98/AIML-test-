import torch
import numpy as np 
import cv2
import pickle
import collections
import os
import time
import sys

from model import *

sys.path.append('../')
from models import fpsCounter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loader(path):
    image = np.asarray(cv2.imread(path)).astype(np.uint8) # [H x W x C, BGR format]
    return image.copy()

def files_inference(weights, data_folder, class_labels, device='cpu'):
	face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	classifier = EmotionNet(5)
	with open(weights, "rb") as weightfile:
		data = pickle.load(weightfile)
		data = collections.OrderedDict(data)
		classifier.load_state_dict(data)
	classifier.eval()
	try:
		files = [f for f in os.listdir(data_folder)]
	except:
		print("No such file or directory exists %s" %data_folder)
		return
	inference_folder = os.path.join("./data", "inference")
	if not os.path.exists(inference_folder):
		os.makedirs(inference_folder)
	invalid_files = []
	for file in files:
		try:
			sample = loader(os.path.join(data_folder, file))
		except:
			invalid_files.append(os.path.join(data_folder, file))
			continue
		labels = []
		gray = sample.copy()
		if len(gray.shape) == 3 and gray.shape[-1] != 1:
			gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
		faces = face_classifier.detectMultiScale(gray,1.3,5)
		for (x, y, w, h) in faces:
			cv2.rectangle(sample, (x,y), (x+w,y+h), (255,0,0), 2)
			roi = gray[y:y+h, x:x+w]
			roi = cv2.resize(roi, (48,48), interpolation=cv2.INTER_AREA)

			if np.sum([roi]) != 0:
				roi = roi.astype('float')/255
				roi = torch.from_numpy(roi.copy()).unsqueeze(0).unsqueeze(0)
				roi = roi.type(torch.FloatTensor).to(device)
				roi = (roi - 0.5076) / 0.0647
				with torch.no_grad():
					pred = classifier(roi).squeeze()
				_, ind = torch.max(pred, dim=0)
				label = class_labels[ind.item()]
				label_position = (x,y)
				cv2.putText(sample, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
			else:
				cv2.putText(sample, 'No Face Found', (20,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
		cv2.imwrite(os.path.join(inference_folder, file), sample)
	if len(invalid_files) > 0:
		print("The following files %d could not be processed:" %(len(invalid_files)))
		for i in range(len(invalid_files)):
			print(invalid_files[i])

def camfeed_inference(weights, class_labels, cameraName, cameraDetails, device = 'cpu'):
	face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	classifier = EmotionNet(5)
	with open(weights, "rb") as weightfile:
		data = pickle.load(weightfile)
		data = collections.OrderedDict(data)
		classifier.load_state_dict(data)
	classifier.eval()
	flag = False
	genders = ["Male", "Female"]
	genderProto = "./gender_detector/gender_deploy.prototxt"
	genderModel = "./gender_detector/gender_net.caffemodel"
	genderNet = cv2.dnn.readNet(genderModel, genderProto)
	# genders, genderNet = genderDetection()
	
	ageList = ['(0-3)', '(4-7)', '(8-13)', '(14-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
	ages = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(21-24)", "(25-32)", "(33-37)", "(38-43)", "(44-47)", "(48-53)", "(54-59)", "(60-100)"]
	ageProto = "./age_detector/age_deploy.prototxt"
	ageModel = "./age_detector/age_net.caffemodel"
	ageNet = cv2.dnn.readNet(ageModel, ageProto)
	
	MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
	try:
		cap = cv2.VideoCapture()
		print(f"Connecting to {cameraName}")
		cap.open(cameraDetails)
		# cap = cv2.VideoCapture(1)
		previousTime = 0
		while True:
			ret, frame = cap.read()
			labels = []
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			faces = face_classifier.detectMultiScale(gray,1.3,5)
			for (x, y, w, h) in faces:
				cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
				roi = gray[y:y+h, x:x+w]
				roi = cv2.resize(roi, (48,48), interpolation=cv2.INTER_AREA)
				blob = cv2.dnn.blobFromImage(frame, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB = False)
				
				genderNet.setInput(blob)
				ageNet.setInput(blob)
				if np.sum([roi]) != 0:
					roi = roi.astype('float')/255
					roi = torch.from_numpy(roi.copy()).unsqueeze(0).unsqueeze(0)
					roi = roi.type(torch.FloatTensor).to(device)
					roi = (roi - 0.5076) / 0.0647
					with torch.no_grad():
						pred = classifier(roi).squeeze()
					_, ind = torch.max(pred, dim=0)
					label = class_labels[ind.item()]
					label_position = (x,y)
					genderPosition = (x, y+20)
					agePosition = (x, y-40)
					cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
					genderPreds = genderNet.forward()
					gender = genders[genderPreds[0].argmax()]
					agePreds = ageNet.forward()
					age = ages[agePreds[0].argmax()]
					currentTime = time.time()
					fps = 1/(currentTime - previousTime)
					previousTime = currentTime
					print("Gender: {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
					fpsCounter(frame, fps)
					cv2.putText(frame, f"Gender: {gender}", genderPosition, cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
					cv2.putText(frame, f"Age: {age}", agePosition, cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

				else:
					cv2.putText(frame, 'No Face Found', (20,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
			cv2.imshow(cameraName, frame)
			if cv2.waitKey(5) & 0xFF == ord('q'):
				flag = True
				break
		cap.release()
		cv2.destroyAllWindows()
		
			
	except:
		print("Unexpected error!\n")
		if not flag:
			cap.release()
			cv2.destroyAllWindows()
			raise

	
    