import warnings
warnings.filterwarnings("ignore")

# In[2]:





# In[3]:


import os
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
from PIL import Image
from keras import backend as K
from keras.models import load_model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_eval
import cv2
import argparse

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
yolo_model = load_model("model_data/yolo.h5")
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
sess = K.get_session()


cont = 1

while(cont == 1):
    print('Enter path:')
    pth = input()
    img = cv2.imread(pth)
    height,width = img.shape[:2]
    width = np.array(width, dtype=float)
    height = np.array(height, dtype=float)
    image_shape = (height, width)
    boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
    image, image_data = preprocess_image(pth, model_image_size = (608, 608))
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
    print('Found',len(out_boxes),'boxes')
    colors = generate_colors(class_names)
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    image.show()
    cont = int(input())
