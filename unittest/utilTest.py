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

from mod.util import preprocess_fn, draw_outputs

class UTIL_TEST_CASE(unittest.TestCase):
    def setUp(self):
        with open("verify/preprocess_fn_in", "rb") as file:
            self.frame = pickle.load(file)
    
    def test_preprocess_fn(self):
        rs = [416, 416]

        with open("verify/preprocess_fn_out", "rb") as file:
            output = pickle.load(file)

        result  = preprocess_fn(self.frame, rs)

        self.assertEqual(np.array_equal(result, output), True, "Incorrect Result.")

    def test_draw_outputs(self):
        total_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motobike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv']
        scores = np.array([[0.9991943, 0.9801216, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)
        classes = np.array([[11, 6, 14, 14, 14, 14, 14, 14, 14, 14]], dtype=int)
        nums = np.array([2], dtype=int)
        
        with open("verify/draw_outputs_boxes", "rb") as file:
            boxes = pickle.load(file)

        with open("verify/draw_outputs_out", "rb") as file:
            output = pickle.load(file)

        result = draw_outputs(self.frame, (boxes, scores, classes, nums), total_classes, 0, (0, 0, 255), 0)

        self.assertEqual(np.array_equal(result, output), True, "Incorrect Result.")
        

if __name__ == '__main__':
    unittest.main()