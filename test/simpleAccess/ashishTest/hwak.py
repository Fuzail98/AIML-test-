import cv2
import numpy as np
from rich import print
from pathlib import Path

from ultralytics import YOLO

from folder.videoStream import VideoStream
from folder.config_checker import Config_monitor


class Hwak:
	def __init__(self, cfg, device, detector_weights, gender_weights):
		self.cfg = cfg
		self.device = device
		self.detector_weights = detector_weights
		self.img_w = 455
		self.img_h = 256
		self.canvas_h = 768
		self.canvas_w = 1366 #canvas size is the monitor resolution
		self.canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

		self.draw_bbox = True
		self.draw_centroid = True
		self.show_gender = False

		#. Load detector
		# self.detector = YOLO(detector_weights)

		#.Load gender detector
		# self.gender_detector = YOLO(gender_weights)

		#.Load Tracker
		#loaded in camera.py

		#.Start a thread to monitor config file
		self.config_monitor = Config_monitor(cfg)

	def check_for_config_updates(self):
		''' Check for config updates and parse the updated config '''
		if self.config_monitor.updated:
			print('[bold #55ff55] Config updated [/bold #55ff55] :star: \n')
			cv2.destroyAllWindows()
			self.canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

			
			#. Parse config
			cam_sources, ROIs, POIs, models = self.config_monitor.parse_config()

			#. Create a VideoStream object for each camera in the config file
			self.video_streams = VideoStream(sources=cam_sources, img_h=self.img_h, img_w=self.img_w, ROIs=ROIs, POIs=POIs, models=models, device=self.device)  #.{cam_id: camera_object}

			self.config_monitor.updated = False
			print('[bold #55ff55] Config Parsed [/bold #55ff55] :thumbsup: \n')

	def generate_model_streams(self, cameras):
		''' split camera streams into streams for each model {model: {cam_id: camera_object}} '''
		models_cameras = {'det': {}, 'track': {}, 'gender': {}, 'weapon': {}}  #.{model: {cam_id: camera_object}}
		for cam_id, camera in cameras.items():
			for model in camera.models:
				models_cameras[model][cam_id] = camera
		return models_cameras

	def draw_results(self, image, outputs):
		''' Draw bounding boxes and centroids '''
		cs = []
		for i, data in enumerate(outputs[0].boxes.data):
			data = data.cpu().numpy()
			x0,y0,x1,y1,score,cls_id = data
			if score > 0 and cls_id == 0: 
				cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (255,255,0), 2)
				centroid = (int((x0+x1)/2), int(y1))
				cs.append(centroid)
		return image, cs

	# def tile_frames(self, streams, num_cams):
	# 	''' Concatenate 4 streams into 1 frame '''
	# 	frames = []
	# 	#. Add extra cameras to make it divisible by 4
	# 	num_cams = num_cams + 4 - num_cams % 4
	# 	streams += [np.ones((self.img_h, self.img_w, 3), dtype=np.uint8)*255 for i in range(num_cams-len(streams))]
	# 	for i in range(num_cams//4):
	# 		top = np.concatenate([streams[i*4+j]  for j in range(2)], axis=0)
	# 		bottom = np.concatenate([streams[i*4+j]  for j in range(2,4)], axis=0)
	# 		frames.append(np.concatenate([top, bottom], axis=1))
	# 	return frames

	def visualize_results(self, cameras):
		''' Visualize results and find centroid trace'''

		#. Calculate the number of rows and columns for the canvas based on the number of frames
		num_frames = len(cameras)
		num_rows = int(np.ceil(np.sqrt(num_frames)))
		num_cols = int(np.ceil(num_frames / num_rows))
		tile_width = int(self.canvas_w / num_cols)
		tile_height = int(self.canvas_h / num_rows)

		#. Iterate over each frame and draw bbox, centroid and place it on the canvas		
		for i,(cam_id, camera) in enumerate(cameras.items()):
			camera.out_frame = cv2.cvtColor(camera.out_frame, cv2.COLOR_RGB2BGR)
			frame_centroids = []
			# if 'track' not in camera.models:
			# 	for bbox in camera.bboxes:
			# 		x0,y0,x1,y1,score,cls_id = bbox
					
			# 		#. Show gender
			# 		if self.show_gender and 'gender' in camera.models:
			# 			gender = 'M' if cls_id == 1 else 'F'
			# 			color = [255,255,0] if cls_id == 1 else [255,0,255]
			# 			camera.out_frame = cv2.putText(camera.out_frame, gender, (int(x1), int(y0)), 0, 5e-3 * 200, color, 2)
			# 		else:
			# 			color = [0,255,255]
					
			# 		#. Draw bboxes
			# 		if self.draw_bbox:
			# 			camera.out_frame = cv2.rectangle(camera.out_frame, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
			# 		centroid = (int((x0+x1)/2), int(y1))
			# 		frame_centroids.append(centroid)
					
			if 'track' in camera.models:
				for bbox in camera.tracked_bboxes:
					x0,y0,x1,y1,track_id,score,cls_id = bbox
					
					#. Show gender
					if self.show_gender and 'gender' in camera.models:
						gender = 'M' if cls_id == 1 else 'F'
						color = [255,255,0] if cls_id == 1 else [255,0,255]
						camera.out_frame = cv2.putText(camera.out_frame, gender, (int(x1), int(y0)), 0, 5e-3 * 200, color, 2)
					else:
						color = [0,255,255]
					
					#. Draw bboxes
					if self.draw_bbox:
						camera.out_frame = cv2.rectangle(camera.out_frame, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
					centroid = (int((x0+x1)/2), int(y1))
					frame_centroids.append(centroid)

					#. Show track_id
					camera.out_frame = cv2.putText(camera.out_frame, str(track_id), (int(x0), int(y0)), 0, 5e-3 * 200, color, 2)
			
			#. Append frame centroids to camera centroids
			camera.centroid_trace.append(frame_centroids)
			if len(camera.centroid_trace) > 50:
				camera.centroid_trace.pop(0)
			
			#. Draw centroid trace
			if self.draw_centroid:
				for centroids in camera.centroid_trace:
						for centroid in centroids:
							camera.out_frame = cv2.circle(camera.out_frame, centroid, 2, (0,0,255), -1)

			#. Add the frame to fit the canvas
			resized_frame = cv2.resize(camera.out_frame, (tile_width, tile_height))
			row = i // num_cols
			col = i % num_cols
			x = col * tile_width
			y = row * tile_height
			self.canvas[y:y+tile_height, x:x+tile_width] = resized_frame
		return self.canvas

	def tile_frames(self, cameras):
		# Calculate the number of rows and columns based on the number of frames
		num_frames = len(cameras)
		num_rows = int(np.ceil(np.sqrt(num_frames)))
		num_cols = int(np.ceil(num_frames / num_rows))

		# Create a blank canvas with the size of the monitor (1920x1080)
		canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

		# Calculate the width and height of each tile
		tile_width = int(self.canvas_w / num_cols)
		tile_height = int(self.canvas_h / num_rows)

		# Iterate over each frame and place it on the canvas
		for i, camera in enumerate(cameras.items()):
			# Resize the frame to fit the tile size
			resized_frame = cv2.resize(camera.out_frame, (tile_width, tile_height))

			# Calculate the row and column indices for the current frame
			row = i // num_cols
			col = i % num_cols

			# Calculate the coordinates to place the frame on the canvas
			x = col * tile_width
			y = row * tile_height

			# Place the resized frame on the canvas
			canvas[y:y+tile_height, x:x+tile_width] = resized_frame

		return canvas

			
	def track_people(self, frames, ROIs):
		''' Track people '''
		out_frames = []
		for i, (frame, ROI) in enumerate(zip(frames, ROIs)):
			dets = self.detector(frame,verbose=False, classes=[0])[0].boxes.data.cpu().numpy()
			tracker_outputs = self.trackers[i+1].update(dets, frame) 
			print('tracker_outputs: ', tracker_outputs)
			for x1, y1, x2, y2, track_id, conf, cls in tracker_outputs:
				color = self.colors[int(track_id)]
				cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
				cv2.putText(frame, str(track_id), (int(x1), int(y1)), 0, 5e-3 * 200, color, 2)
			out_frames.append(frame)
		return out_frames

