#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings


# In[2]:


warnings.filterwarnings("ignore")


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


# In[4]:


ap = argparse.ArgumentParser()


# In[5]:


ap.add_argument('-i','--image',required = True,help = 'image location')


# In[6]:


args = vars(ap.parse_args())


# In[32]:


img = cv2.imread(args['image'])


# In[34]:


height,width = img.shape[:2]
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)


# In[35]:


image_shape = (height, width)


# In[36]:


class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")


# In[11]:


yolo_model = load_model("model_data/yolo.h5")


# In[12]:


yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))


# In[37]:


boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# In[38]:


sess = K.get_session()


# In[39]:


image, image_data = preprocess_image(args['image'], model_image_size = (608, 608))


# In[40]:


out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})


# In[41]:


print('Found',len(out_boxes),'boxes')


# In[42]:


colors = generate_colors(class_names)


# In[43]:


draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)


# In[44]:

pth = args['image'].split(sep = '.')[0]
print(pth)
image.save((pth+'-op.jpg'), quality=90)


# In[ ]:
image.show()


# In[ ]:




