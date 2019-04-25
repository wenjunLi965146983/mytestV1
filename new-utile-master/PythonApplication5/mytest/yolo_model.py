
from keras import models
from keras.layers import Lambda,Input

import keras.backend as K
import tensorflow as tf
import numpy as np


import ModelTool as MT

'''
Define
'''
anchors=[[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]]



'''
model 
'''


def create_model(input_shape,anchors,num_classes):
    '''create the training model'''

    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
   
    w, h = input_shape
    num_anchors=len(anchors)
    y_true=[Input(shape=(w//{0:32, 1:16, 2:8}[i],h//{0:32, 1:16, 2:8}[i],num_anchors,num_classes+5)) for i in range(num_anchors//3)]


    yolo_body=MT.yolo_model(image_input,num_anchors,num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))



    model_loss= Lambda(MT.yolo_loss,output_shape=(1,),name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*yolo_body.output, *y_true])


    return models.Model([image_input,y_true],model_loss)



'''
layers
'''


