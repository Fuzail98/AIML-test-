
import sys
import torch
import time
import cv2
import argparse
import time
import numpy as np
from rich import print

from hwak import Hwak


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--cfg', default='configs/config.yaml', help='config file')
	parser.add_argument('-d', '--device', default='cuda', help='device')
	parser.add_argument('-dw', '--detector_weights', default='yolov8n.pt', help='detector_weights')
	parser.add_argument('-gw', '--gender_weights', default='gender_det.pt', help='gender_weights')
	return parser.parse_args()


print('[bold green] ----------------- Starting  ----------------- [/bold green]\n')
args = parse_args()
hwak = Hwak(args.cfg, args.device, args.detector_weights, args.gender_weights)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)

while True:
	start = time.time()

	#. Parse config and check for updates and parse the updated config
	hwak.check_for_config_updates()
	
	#. Read streams
	cameras = hwak.video_streams.read()  #.{cam_id: camera_object}
	
	#. print camera status
	for cam_id, camera in cameras.items():
		camera.print_status()

	for cam_id, camera in cameras.items():
		frame = camera.frame
		camera.out_frame = frame.copy()
		#. Detect people  	
		if 'det' in camera.models:
			camera.bboxes = hwak.detector(frame,verbose=False, classes=[0])[0].boxes.data.cpu().numpy()

		#. Detect gender
		if 'gender' in camera.models:
			camera.bboxes = hwak.gender_detector(frame,verbose=False, classes=[0,1])[0].boxes.data.cpu().numpy()

		#. Track people					
		if 'track' in camera.models:
			camera.tracked_bboxes = camera.tracker.update(camera.bboxes, frame) 

		

	#. Display
	canvas = hwak.visualize_results(cameras)
	canvas = cv2.putText(canvas, 'camera', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
	cv2.imshow('Output', canvas)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		sys.exit(0)
	



