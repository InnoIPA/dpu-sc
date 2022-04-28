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

from mod.predictor import PREDICTOR
from mod.dpu import YOLOV3, XMODEL

class YOLO_TEST_CASE(unittest.TestCase):
    def setUp(self):
        with open("verify/frame", "rb") as file:
            frame = pickle.load(file)
        args = ""
        cfg = {'MODLES': {'XMODELS_CLASS': {'TYPE': 'cnn',
                                            'MODEL': 'models/cnn/customcnn.xmodel',
                                            'CLASS': ['dog', 'cat'],
                                            'INPUT_SIZE': [250, 200]}}}
        
        self.frame = frame
        self.pred = PREDICTOR(args, cfg)

    @mock.patch.object(PREDICTOR, 'runDPU_') 
    @mock.patch.object(XMODEL, 'init')
    def test_run_yolo(self, mock_init, mock_runDPU_):
        with open("verify/cnn_out", "rb") as file:
            runDPU_output = pickle.load(file)

        mock_init.return_value = None
        mock_runDPU_.return_value = runDPU_output

        self.pred.init_cnn()
        _ = self.pred.run_cnn(self.frame)

        prediction = self.pred.classes[np.argmax(self.pred.x.outputs[0][0])]

        self.assertEqual(prediction, 'dog', "Incorrect Result.")
        

if __name__ == '__main__':
    unittest.main()