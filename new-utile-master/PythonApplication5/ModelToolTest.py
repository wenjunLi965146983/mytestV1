from keras.applications import vgg16
from keras import models
from keras import layers

import keras.backend as K
import tensorflow as tf
import numpy as np
from ModelTollTest import feature_map

'''
model 
'''


def test():

    

    K.clear_session() # get a new session
    model=vgg16.VGG16(include_top=False,weights='imagenet',input_shape=(448,448,3))

    for layer in model.layers:
        layer.trainable=False



    feature_map()






if __name__== 'main':
    test()