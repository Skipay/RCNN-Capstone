import cv2
from visualize_cv2 import model, display_instances, class_names
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import skimage.io
import math
import sys
import os

args = sys.argv
if(len(args)<2):
    print("run command: python video_demo,py 0 or video file name")

name = args[1]
if (args[1] == '0'):
    name = int(args[1])

stream = cv2.VideoCapture(name)
while True:
    ret , frame = stream.read()
    if not ret:
        print("unable to fetch frame")
        break
    results = model.detect([frame], verbose = 1)

    r = results[0]
    masked_image = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

    cv2.imshow("masked_image", masked_image)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
stream.release()
