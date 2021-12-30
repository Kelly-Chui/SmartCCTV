import cv2
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import argparse
import serial
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='detect')
    parser.add_argument('--fps', dest='fps', default=20, type=int)
    parser.add_argument('--width', dest='width', default=640, type=int)
    parser.add_argument('--height', dest='height', default=480, type=int)
    args = parser.parse_args()
    return args

args = parse_args()
cam0 = 0
cam1 = 1

saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
print("preprocessing")


cap, fback = camera_init(cam0)
# cam2, fback2 = camera_init(cam1)

li = []
li2 = []
# 비디오 매 프레임 처리
while True:

    frame1, aa = camera_read(cap, fback, saved_model_loaded, name=str(cam0))
    # frame2 = camera_read(cam2, fback2, saved_model_loaded, name=str(cam1))

    li.append(frame1)
    # li2.append(frame2)
    k = cv2.waitKey(1)
    if k == 27:
        break

video_writing(args.fps, li, cap, args.width, args.height, "cam0")


# video_writing(args.fps, li2, cam2, args.width, args.height, "cam1")


