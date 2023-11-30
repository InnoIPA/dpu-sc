# Copyright (c) 2022 Innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import cv2
import numpy as np
import pickle
import unittest
from unittest import mock
import sys
sys.path.append("..")
import argparse

from mod.predictor import PREDICTOR
from mod.dpu import YOLOV3, XMODEL

class YOLO_TEST_CASE(unittest.TestCase):
    def setUp(self):
        with open("verify/frame", "rb") as file:
            frame = pickle.load(file)
        args = ""
        cfg = {'MODLES': {'XMODELS_OBJ': {'TYPE': 'yolo',
                                          'MODEL': 'models/obj/yolov3_voc_416_v25_d3136.xmodel:',
                                          'CLASS': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motobike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'],
                                          'ANCHORS': [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                                          'INPUT_SIZE': [416, 416],
                                          'IOU': '0.213',
                                          'NMS': '0.45',
                                          'CONF': '0.3',
                                          'BOX_MAX_NUM': '10',
                                          'LABEL_CLASSES': '20'}}}
        
         
        self.frame = frame
        self.pred = PREDICTOR(args, cfg)

    @mock.patch.object(PREDICTOR, 'runDPU_') 
    @mock.patch.object(XMODEL, 'init')
    def test_run_yolo(self, mock_init, mock_runDPU_):
        ap = argparse.ArgumentParser()
        ap.add_argument('-i', '--image' , type=str)
        args = ap.parse_args()
        args.image = "1"
        cfg = {'MODLES': {'XMODELS_OBJ': {'TYPE': 'yolo',
                                          'MODEL': 'models/obj/yolov3_voc_416_v25_d3136.xmodel:',
                                          'CLASS': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motobike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'],
                                          'ANCHORS': [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                                          'INPUT_SIZE': [416, 416],
                                          'IOU': '0.213',
                                          'NMS': '0.45',
                                          'CONF': '0.3',
                                          'BOX_MAX_NUM': '10',
                                          'LABEL_CLASSES': '20'}}}
        self.pred = PREDICTOR(args, cfg)
        with open("verify/sort_boxes_in", "rb") as file:
            runDPU_output = pickle.load(file)

        with open("verify/pred_boxes_boxes", "rb") as file:
            pred_boxes_boxes = pickle.load(file)
        with open("verify/pred_boxes_scores", "rb") as file:
            pred_boxes_scores = pickle.load(file)
        with open("verify/pred_boxes_classes", "rb") as file:
            pred_boxes_classes = pickle.load(file)
        with open("verify/pred_boxes_nums", "rb") as file:
            pred_boxes_nums = pickle.load(file)
        
        with open("verify/yolo_out", "rb") as file:
            output = pickle.load(file)

        mock_init.return_value = None
        mock_runDPU_.return_value = runDPU_output

        self.pred.init_yolo()
        image = np.asarray(self.frame)
        cv2.imwrite("tmp.jpg", image)
        frame = []
        frame = cv2.imread('./tmp.jpg')
        result = self.pred.run_yolo(frame)

        flag1 = np.array_equal(np.around(self.pred.p_boxes, 4), np.around(pred_boxes_boxes, 4))
        flag2 = np.array_equal(np.around(self.pred.p_scores, 4), np.around(pred_boxes_scores, 4))
        flag3 = np.array_equal(self.pred.p_classes, pred_boxes_classes)
        flag4 = np.array_equal(self.pred.p_nums, pred_boxes_nums)

        self.assertEqual((flag1 & flag2 & flag3 & flag4), True, "Incorrect Result.")
        

if __name__ == '__main__':
    unittest.main()
