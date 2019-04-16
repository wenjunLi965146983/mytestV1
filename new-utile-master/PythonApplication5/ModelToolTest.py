from keras.applications import vgg16
from keras import models
from keras import layers
import keras.backend as K
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt



import ModelTool as t
import utile as lwj


'''
model 
'''


def test():

    

    K.clear_session() # get a new session
    model=vgg16.VGG16(include_top=True,weights='imagenet',input_shape=(224,224,3))

    for layer in model.layers:
        layer.trainable=False

    model.summary()


    generater=lwj.datagenerater("E:/VOC2012/JPEGImages/","E:/VOC2012/Annotations/",batch_size=1,image_size=(224,224))
    datagenerater=generater.get_train_data()
    
    keep_run=True
    while keep_run:
        image,label=next(datagenerater)
        tt=np.reshape(image,(224,224,3))
        plt.imshow(tt)
        plt.show()
        image=image/255
        t.feature_map(model,image,[-1,0])
        a=input('input:')
        if a=='l':
            keep_run=False

    
   
  




if __name__== '__main__':
    test()