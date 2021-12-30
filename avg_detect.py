import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
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

def parse_args():
    parser = argparse.ArgumentParser(description='detect')
    parser.add_argument('--framework', dest='framework', default='tf', type=str)
    parser.add_argument('--weights', dest='weights', default='./checkpoints/yolov4-416', type=str)
    parser.add_argument('--size', dest='size', default=416, type=int)
    parser.add_argument('--tiny', dest='tiny', default=False, type=bool)
    parser.add_argument('--model', dest='model', default='yolov4', type=str)
    parser.add_argument('--image', dest='image', default='./data/kite.jpg', type=str)
    parser.add_argument('--output', dest='output', default='result.png', type=str)
    parser.add_argument('--iou', dest='iou', default=0.45, type=float)
    parser.add_argument('--score', dest='score', default=0.25, type=float)

    
    args = parser.parse_args()
    return args


def main(img, saved_model_loaded):
    args = parse_args()
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    input_size = args.size
    args.image = img
    image_path = args.image

    original_image = args.image

    # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    # image_data = image_data[np.newaxis, ...].astype(np.float32)

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)


    # saved_model_loaded = tf.saved_model.load(args.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=args.iou,
        score_threshold=args.score
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image = utils.draw_bbox(original_image, pred_bbox)
    # image = utils.draw_bbox(image_data*255, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))
    # image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    # return image
    # cv2.imwrite(args.output, image)

if __name__ == '__main__':
    img = cv2.imread('./data/kite.jpg')
    main(img, tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING]))
