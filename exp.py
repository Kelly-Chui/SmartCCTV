import cv2
import sys
import numpy as np
from avg_detect import main
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import argparse
from utils import *
# 비디오 파일 열

fps = 20
width = 640
height = 480
cam0 = 0
cam1 = 1

saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
print("preprocessing")

# cap = cv2.VideoCapture('./data/cctv.mp4')

cap, fback = camera_init(cam0)
cam2, fback2 = camera_init(cam1)

li = []
li2 = []
# 비디오 매 프레임 처리
while True:
    frame1 = camera_read(cap, fback, saved_model_loaded, name=str(cam0))
    frame2 = camera_read(cam2, fback2, saved_model_loaded, name=str(cam1))
    li.append(frame1)
    li2.append(frame2)
    k = cv2.waitKey(1)
    if k == 27:
        break

video_writing(fps, li, cap, width, height, "cam0")
video_writing(fps, li2, cam2, width, height, "cam1")


