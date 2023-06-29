# Copyright (c) 2022 Innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import cv2
import numpy as np
import logging
import json
import time

# try:
import tensorflow as tf
# except:
# 	pass

def preprocess_fn(image, rs):
	'''
	Image pre-processing.
	Rearranges from BGR to RGB then normalizes to range 0:1
	input arg: path of image file
	return: numpy array
	'''
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, rs, cv2.INTER_LINEAR)
	image = image/255.0
	return image

def draw_outputs(img, outputs, class_names, i, color, fps):
	boxes, objectness, classes, nums = outputs
	boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
	wh = np.flip(img.shape[0:2])
	
	x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
	x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
	img = cv2.rectangle(img, x1y1, x2y2, color, 2)
	img = cv2.putText(img, '{} {:.4f}'.format(class_names[int(classes[i])], objectness[i]), x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
	img = cv2.putText(img, 'fps: {:.2f}'.format(fps), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
			
	return img

def draw_outputs_lpr(self,img, outputs, class_names, i, color, time1):
	boxes, objectness, classes, nums = outputs
	boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
	wh = np.flip(img.shape[0:2])
	x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
	x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
	Cropped = img[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]]
	if Cropped.size != 0:
		Cropped_img = load_lpr_img(Cropped)
		prebs = self.runDPU_LPR(Cropped_img)
		pred_text = gd_code(prebs)

	time2 = time.time()
	time_total = time2 - time1
	fps = 1 / time_total

	if Cropped.size != 0:
		img = cv2.rectangle(img, x1y1, x2y2, color, 2)
		img = cv2.putText(img, '{}'.format(pred_text), x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, color, 2)
		img = cv2.putText(img, 'fps: {:.2f}'.format(fps), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
			
	return fps, time_total, img,pred_text

def load_lpr_img(Cropped):
	#load圖片
	img = cv2.resize(Cropped,(94,24))
	img = img.astype('float32')
	img -= 127.5
	img *= 0.0078125
	img = np.expand_dims(img,axis=0)
	return img

def gd_code(prebs):
	#LPRNet Greedy_Decode
	CHARS = [
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                'W', 'X', 'Y', 'Z', 'I', 'O', '-'
                ]

	preb_labels = list()
	for i in range(prebs.shape[0]):
		preb = prebs[i, :, :]
		preb_label = list()
		for j in range(preb.shape[1]):
			preb_label.append(np.argmax(preb[:, j], axis=0))
		no_repeat_blank_label = list()
		pre_c = preb_label[0]
		if pre_c != len(CHARS) - 1:
			no_repeat_blank_label.append(pre_c)
		for c in preb_label: # dropout repeate label and blank label
			if (pre_c == c) or (c == len(CHARS) - 1):
				if c == len(CHARS) - 1:
					pre_c = c
				continue
			no_repeat_blank_label.append(c)
			pre_c = c
		preb_labels.append(no_repeat_blank_label)
	for i, label in enumerate(preb_labels):
		lb = ""
		for i in label:
			lb += CHARS[i]

	return lb

def logging():
    console = logging.StreamHandler()
    logging.getLogger().setLevel(logging.NOTSET)
    # consoleformatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    consoleformatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console.setFormatter(consoleformatter)
    logging.getLogger().addHandler(console)

    return logging.getLogger()

def open_labels(ladels_path):
	with open(ladels_path, 'r') as label_file:
		__labels = [l.strip() for l in label_file.readlines()]
	return __labels

def open_json(CFG):
	with open(CFG) as json_file:
		cfg = json.load(json_file)
	return cfg