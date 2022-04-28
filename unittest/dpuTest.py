# Copyright (c) 2022 Innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import pickle
import unittest
from unittest import mock
import sys
import numpy as np
sys.path.append("..")

from mod.predictor import PREDICTOR
from mod.dpu import YOLOV3

class DPU_TEST_CASE(unittest.TestCase):
    def setUp(self):
        with open("verify/frame", "rb") as file:
            frame = pickle.load(file)
        with open("verify/sort_boxes_in", "rb") as file:
            self.input = pickle.load(file)
        with open("verify/sort_boxes_out", "rb") as file:
            self.sorted = pickle.load(file)
        
        args = ""
        anchors = np.array([[0.02403846, 0.03125],
                            [0.03846154, 0.07211538],
                            [0.07932692, 0.05528846],
                            [0.07211538, 0.14663462],
                            [0.14903846, 0.10817308],
                            [0.14182692, 0.28605769],
                            [0.27884615, 0.21634615],
                            [0.375     , 0.47596154],
                            [0.89663462, 0.78365385]])
                            
        # label_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motobike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv']

        
        self.frame = frame
        self.yolo = YOLOV3('', '', 416, 10, anchors, 20, 0.213, 0.45, 0.3)

    def test_sort_boxes(self):
        self.yolo.outputs = self.input
        result = self.yolo.sort_boxes()

        self.assertEqual(np.array_equal(result, self.sorted), True, "Incorrect Result.")
    
    def test_pred_boxes(self):
        self.yolo.outputs = self.input
        self.yolo.sorted = self.sorted

        result = self.yolo.pred_boxes()

        with open("verify/pred_boxes_boxes", "rb") as file:
            pred_boxes_boxes = pickle.load(file)
        with open("verify/pred_boxes_scores", "rb") as file:
            pred_boxes_scores = pickle.load(file)
        with open("verify/pred_boxes_classes", "rb") as file:
            pred_boxes_classes = pickle.load(file)
        with open("verify/pred_boxes_nums", "rb") as file:
            pred_boxes_nums = pickle.load(file)

        boxes, scores, classes, nums = self.yolo.pred_boxes()

        flag1 = np.array_equal(np.around(boxes, 4), np.around(pred_boxes_boxes, 4))
        flag2 = np.array_equal(np.around(scores, 4), np.around(pred_boxes_scores, 4))
        flag3 = np.array_equal(classes, pred_boxes_classes)
        flag4 = np.array_equal(nums, pred_boxes_nums)

        self.assertEqual((flag1 & flag2 & flag3 & flag4), True, "Incorrect Result.")


if __name__ == '__main__':
    unittest.main()