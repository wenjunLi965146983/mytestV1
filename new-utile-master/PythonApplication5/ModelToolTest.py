from keras.applications import vgg16
from keras.applications import imagenet_utils
from keras import models
from keras import layers
import keras.backend as K
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from keras.applications import resnet50


import ModelTool as t
import utile as lwj
import ImageNetObjectClass as INOC


'''
model 
'''


def test():

    

    K.clear_session() # get a new session
    model=resnet50.ResNet50(include_top=True,weights='imagenet',input_shape=(224, 224, 3))

    for layer in model.layers:
        layer.trainable=False

    model.summary()


    generater=lwj.datagenerater("E:/VOC2012/JPEGImages/","E:/VOC2012/Annotations/",batch_size=1,image_size=(224,224))
    datagenerater=generater.get_train_data()
    

    #object=INOC.read_json('E:/github/imagenet_1000_labels.json')

    keep_run=True
    while keep_run:
        image,label=next(datagenerater)
        
        image=image
        tt=np.reshape(image,(224,224,3))
        plt.imshow(tt)
        plt.show()
        image=np.reshape(image,(1,224,224,3))
        activations = model.predict(image) 
        print(resnet50.decode_predictions(activations))
       
      
    
       
     
    #    print(imagenet_utils.decode_predictions(activations))

       
       
        
    
   
  




if __name__== '__main__':
    test()