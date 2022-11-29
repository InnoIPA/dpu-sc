# Copyright (c) 2022 Innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import cv2
import time
import numpy as np
import logging
import random
import colorsys
# from mod.camera import *

# try:
#     from vimba import *
# except:
# 	pass

try:
    from mod.camera import *
except:
    pass

from mod.dp import DP
from mod.util import preprocess_fn
from mod.util import draw_outputs
from mod.dpu import XMODEL, YOLOV3
# from mod.util import open_json ,open_labels



import pickle

ID          = 'ID'
IOU         = 'IOU'
NMS         = 'NMS'
CONF        = 'CONF'
MODLES      = 'MODLES'
TYPE        = 'TYPE'
MODEL       = 'MODEL'
CLASS       = 'CLASS'
OUTPUT      = 'OUTPUT'
G4_CAMERA   = 'G4_CAMERA'
FW_PATH     = 'FW_PATH'
CLASS_NAME  = "CLASS_NAME"
ANCHORS     = "ANCHORS"
INPUT_SIZE  = "INPUT_SIZE"
DISPLAY     = "DISPLAY"
WIDTH       = "WIDTH"
HEIGHT      = "HEIGHT"

BOX_MAX_NUM     = 'BOX_MAX_NUM'
VIDEO_OUTPUT    = 'VIDEO_OUTPUT'
IMAGE_OUT_DIR   = 'IMAGE_OUT_DIR'
LABEL_CLASSES   = 'LABEL_CLASSES'
XMODELS_CLASS   = 'XMODELS_CLASS'
XMODELS_OBJ     = 'XMODELS_OBJ'

divider = '------------------------------------'

class PREDICTOR():
    def __init__(self, args, cfg):
        self.cfg    = cfg
        self.args   = args

        self.init_model = None
        self.run_model  = None
        self.output     = None
        self.get_frame  = None

        self.f_first_run_model    = True
        self.f_first_output       = True
        self.f_first_get_frame    = True



    '''
	Func:	Get Frame
	Input:	None
	Output:	Frame(raw)
	'''
    def image_get(self):
        if self.f_first_get_frame:
            self.f_first_get_frame = False
            frame = cv2.imread(self.args.image)
            ret = True

        else:
            if self.args.target == 'dp':
                frame = cv2.imread(self.args.image)
                ret = True
            else:
                frame = None
                ret = False

        return ret, frame

    def video_get(self):
        if self.f_first_get_frame:
            self.f_first_get_frame = False
            self.cap = cv2.VideoCapture(self.args.video)
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            if not self.cap.isOpened():
                print("Cannot open camera")
                exit()
            
            ret, frame = self.cap.read()

        else:
            ret, frame = self.cap.read()

        return ret, frame

    def cam_get(self):
        if self.f_first_get_frame:
            self.f_first_get_frame = False
            width = int(self.cfg[DISPLAY][WIDTH])
            height = int(self.cfg[DISPLAY][HEIGHT])

            self.cap = cv2.VideoCapture(int(self.args.camera))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            if not self.cap.isOpened():
                print("Cannot open camera")
                exit()
            
            ret, frame = self.cap.read()

        else:
            ret, frame = self.cap.read()

        return ret, frame

    # def g4cam_get(self):
    #     __cam_id = self.cfg[G4_CAMERA][ID]
    #     __fw = self.cfg[G4_CAMERA][FW_PATH]

    #     width = int(self.cfg[DISPLAY][WIDTH])
    #     height = int(self.cfg[DISPLAY][HEIGHT])

    #     vimba = VIMBA_CAMERA()
    #     frame_q = queue.Queue()

    #     with Vimba.get_instance():
    #         with vimba.get_camera(__cam_id) as cam:
    #             vimba.setup_camera(cam, __fw)
    #             handler = Handler(frame_q)
    #             try:
    #                 # Start Streaming with a custom a buffer of 10 Frames (defaults to 5)
    #                 cam.start_streaming(handler, buffer_count=10)
    #                 while(True):
    #                     frame = frame_q.get()
    #                     frame = self.run_model(frame)
    #                     frame = cv2.resize(frame, (width, height))
    #                     self.output(frame)

    #             finally:
    #                 cam.stop_streaming()

    '''
	Func:	Init Model
    '''
    def init_yolo(self):
        ''' Get config from cfg '''
        self.iou = float(self.cfg[MODLES][XMODELS_OBJ][IOU])
        self.nms = float(self.cfg[MODLES][XMODELS_OBJ][NMS])
        self.conf = float(self.cfg[MODLES][XMODELS_OBJ][CONF])
        self.box_max_num = int(self.cfg[MODLES][XMODELS_OBJ][BOX_MAX_NUM])
        raw_anchors = self.cfg[MODLES][XMODELS_OBJ][ANCHORS]

        self.model = self.cfg[MODLES][XMODELS_OBJ][MODEL]
        self.classes = self.cfg[MODLES][XMODELS_OBJ][CLASS]
        self.input_size = self.cfg[MODLES][XMODELS_OBJ][INPUT_SIZE]

        self.label_classes = len(self.classes)

        ''' Set anchors '''
        rows = int(len(raw_anchors) / 2)
        cols = 2

        self.anchors = np.zeros(shape=(rows, cols), dtype=float)
        for i in range(rows):
            for j in range(cols):
                self.anchors[i][j] = int(raw_anchors[cols * i + j]) / self.input_size[0]

        # random color for predict box
        self.color_list = []
        for i in range(len(self.classes)):
            h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
            r, g, b = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
            self.color_list.append([r, g, b])

        time_init_start = time.time()
        self.x = YOLOV3(self.model, 'yolov3', self.input_size[0], self.box_max_num, self.anchors, self.label_classes, self.iou, self.nms, self.conf)
        self.x.init()
        logging.debug("Init yolov3 times = {:.4f} seconds".format(time.time() - time_init_start))

    def init_cnn(self):
        ''' Get cfg '''
        self.model = self.cfg[MODLES][XMODELS_CLASS][MODEL]
        self.classes = self.cfg[MODLES][XMODELS_CLASS][CLASS]
        self.input_size = self.cfg[MODLES][XMODELS_CLASS][INPUT_SIZE]

        time_init_start = time.time()
        self.x = XMODEL(self.model, 'cnn')
        self.x.init()
        logging.debug("Init cnn times = {:.4f} seconds".format(time.time() - time_init_start))

    '''
	Func:	Models Select
	Input:	Frame (raw)
	Output:	Frame (inferenced)
    '''
    def run_yolo(self, frame):
        time1 = time.time()
        img = []

        image = preprocess_fn(frame, self.input_size)

        img.append(image)
        self.x.outputs = self.runDPU_(img[0:])

        time_pred_box = time.time()
        
        self.x.sorted = self.x.sort_boxes()
        # self.x.prediction = self.x.pred_boxes()

        logging.debug("bb output times = {:.4f} seconds".format(time.time() - time_pred_box))

        self.p_boxes, self.p_scores, self.p_classes, self.p_nums = self.x.pred_boxes()

        time2 = time.time()
        time_total = time2 - time1

        fps = 1 / time_total

        logging.info(divider)
        logging.info(" Throughput={:.2f} fps, total frames = {:.0f}, time = {:.4f} seconds".format(fps, 1, time_total))

        logging.info(' Detections:')

        for i in range(self.p_nums[0]):
            logging.info('\t{}, {}, {}'.format(self.classes[int(self.p_classes[0][i])],
												np.array(self.p_scores[0][i]),
												np.array(self.p_boxes[0][i])))
            frame = draw_outputs(frame, (self.p_boxes, self.p_scores, self.p_classes, self.p_nums), self.classes, i, self.color_list[int(self.p_classes[0][i])], fps)

        return frame

    def run_cnn(self, frame):
        time1 = time.time()
        img = []
        img.append(preprocess_fn(frame, self.input_size))
        self.x.outputs = self.runDPU_(img[0:])

        prediction = self.classes[np.argmax(self.x.outputs[0][0])]

        time2 = time.time()
        time_total = time2 - time1
        fps = 1 / time_total

        logging.info(divider)
        logging.info("Throughput={:.2f} fps, total frames = {:.0f}, time={:.4f} seconds".format(fps, 1, time_total))

        ''' Put fps and pridict class on image '''
        frame = cv2.putText(frame, '{} fps: {:.4f}'.format(prediction, fps), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

        return frame

    def run_yolo_lpr(self, frame):
        from mod.lpr import draw_outputs_lpr
        
        time1 = time.time()
        img = []

        image = preprocess_fn(frame, self.input_size)

        img.append(image)
        self.x.outputs = self.runDPU_(img[0:])

        time_pred_box = time.time()
        
        self.x.sorted = self.x.sort_boxes()
        # self.x.prediction = self.x.pred_boxes()

        logging.debug("bb output times = {:.4f} seconds".format(time.time() - time_pred_box))

        self.p_boxes, self.p_scores, self.p_classes, self.p_nums = self.x.pred_boxes()

        logging.info(divider)

        logging.info(' Detections:')

        for i in range(self.p_nums[0]):
            logging.info('\t{}, {}, {}'.format(self.classes[int(self.p_classes[0][i])],
												np.array(self.p_scores[0][i]),
												np.array(self.p_boxes[0][i])))
            
            fps, time_total, frame = draw_outputs_lpr(frame, (self.p_boxes, self.p_scores, self.p_classes, self.p_nums), self.classes, i, self.color_list[int(self.p_classes[0][i])], time1)
        
        logging.info(" Throughput={:.2f} fps, total frames = {:.0f}, time = {:.4f} seconds".format(fps, 1, time_total))

        return frame

    '''
	Func:	Output
	Input:	Frame (inferenced)
	Output:	Image, Video or Streaming
	'''
    def dp_out(self, frame):
        if self.f_first_output:
            self.f_first_output = False
            self.dummy_dp = False
            ''' Get cfg '''
            width = int(self.cfg[DISPLAY][WIDTH])
            height = int(self.cfg[DISPLAY][HEIGHT])

            resolution = "{}x{}".format(width, height)
            all_res = os.popen("modetest -M xlnx -c| awk '/name refresh/ {f=1;next}  /props:/{f=0;} f{print $1 \"@\" $2}'").read()

            if all_res.find(resolution) == -1:
                # os.environ["DISPLAY"] = ":0"
                self.dummy_dp = True
                logging.info("Error: Monitor doesn't support resolution {}".format(resolution))
            else:
                self.dp = DP(width, height)

        if self.dummy_dp:
            cv2.imshow("imshow", frame)
            cv2.waitKey(1)
        else:
            self.dp.imshow(frame)

    def image_out(self, frame):
        if self.f_first_output:
            self.f_first_output = False

        output_dir = self.cfg[OUTPUT][IMAGE_OUT_DIR]
        cv2.imwrite('{}o_p_{}.png'.format(output_dir, time.time()), frame)

    def video_out(self, frame):
        if self.f_first_output:
            self.f_first_output = False

            output_file = self.cfg[OUTPUT][VIDEO_OUTPUT]
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            # fps = 4
            self.vw = cv2.VideoWriter(output_file, fourcc, fps, (w, h), True)

        self.vw.write(frame)




    # def check_resolution(self, width, height):
    #     resolution = "{}x{}".format(width, height)
    #     all_res = os.popen("modetest -M xlnx -c| awk '/name refresh/ {f=1;next}  /props:/{f=0;} f{print $1 \"@\" $2}'").read()



    def runDPU_(self, img):
        inputTensors = self.x.get_input_tensors()
        outputTensors = self.x.get_output_tensors()
        
        input_ndim = tuple(inputTensors[0].dims)
        output_ndim = []
        for i in range(len(outputTensors)):
            output_ndim.append(tuple(outputTensors[i].dims))
        
        outputs = [] # Reset output data, if not it will segment fault when run DPU.
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]

        for i in range(len(outputTensors)):
            outputs.append(np.empty(output_ndim[i], dtype=np.float32, order="C"))

        '''init input image to input buffer '''
        imageRun = inputData[0]
        imageRun[0, ...] = img[0].reshape(input_ndim[1:])

        '''init input image to input buffer '''
        '''run with batch '''

        time_pred_start = time.time()
        self.x.predict(inputData, outputs)
        logging.debug("Pred times (DPU function) = {:.4f} seconds".format(time.time() - time_pred_start))
        return outputs

    def predict(self):
        ret, frame = self.get_frame()
        self.init_model()
        
        while ret:
            frame = self.run_model(frame)
            self.output(frame)
            ret, frame = self.get_frame()