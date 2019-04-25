from keras.applications import vgg16
from keras.applications import imagenet_utils
from keras import models
from keras import layers
import keras.backend as K
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from keras.applications import resnet50



import utile as lwj
import ImageNetObjectClass as INOC
import yolo_model

'''
model 
'''


def test():

    

    K.clear_session() # get a new session
    anchors=[[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]]
    anchors=np.array(anchors)
  
    network=yolo_model.create_model((416,416),anchors,2)
    network.summary()
    network.save('myyolotest.h5')
       
       
        
    
   
  




if __name__== '__main__':
    test()