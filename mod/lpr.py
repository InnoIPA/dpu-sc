import re
import cv2
import numpy as np
import pytesseract
import logging
import time

def draw_outputs_lpr(img, outputs, class_names, i, color, time1):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
    x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
    Cropped = img[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]]
    img_gray = cv2.cvtColor(Cropped, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(img_gray)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    text = re.sub(r'[ \n]', '', text)
    logging.info("LPR result: {}".format(text))

    time2 = time.time()
    time_total = time2 - time1
    fps = 1 / time_total

    img = cv2.rectangle(img, x1y1, x2y2, color, 2)
    img = cv2.putText(img, '{}'.format(text), x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
    img = cv2.putText(img, 'fps: {:.2f}'.format(fps), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
            
    return fps, time_total, img