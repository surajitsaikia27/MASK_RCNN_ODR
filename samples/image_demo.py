import tensorflow as tf
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import glob
import argparse

ROOT_DIR = os.path.abspath("../")
os.environ['KERAS_BACKEND'] = 'tensorflow'

sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils

import mrcnn.model as modellib

from mrcnn import visualize

sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

import coco

from samples.coco import coco

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


#IMAGE_DIR='/home/surajit/Desktop/Mask_RCNN-master/my/'

device_name = tf.test.gpu_device_name()
print('Found GPU at: {}'.format(device_name))


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

def parse_args():

    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='MAskRCNN object detection and segmentation')

    parser.add_argument('--path', dest='path', help='provide the path of the image directory',
                        default=0, type=str)

    args = parser.parse_args()

    return args




class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']




if __name__ == '__main__':

    args = parse_args()

    for image in glob.glob(os.path.join(str(args.path),'*.jpg')):
        print(image)
        image = skimage.io.imread(os.path.join(str(args.path), image))

        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])




















