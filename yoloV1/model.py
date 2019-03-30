from keras import models
from keras import layers
from keras import optimizers 
import math
import numpy as np
import keras.backend as tf
class model():
    """description of class"""


    def __init__(self,input_shape):
        self.input_shape=input_shape
    def get_model(self):
        network=models.Sequential()

        network.add(layers.Conv2D(64,(7,7),strides=(2, 2),padding='SAME',input_shape=self.input_shape))
        network.add(layers.MaxPool2D((2,2),strides= 2))

        network.add(layers.Conv2D(192,(3,3),padding='SAME'))
        network.add(layers.MaxPool2D((2,2),strides= 2))

        network.add(layers.Conv2D(128,(1,1),padding='SAME'))
        network.add(layers.Conv2D(256,(3,3),padding='SAME'))
        network.add(layers.Conv2D(256,(1,1),padding='SAME'))
        network.add(layers.Conv2D(512,(3,3),padding='SAME'))
        network.add(layers.MaxPool2D((2,2),strides= 2))


        for i in range(0,4):
            network.add(layers.Conv2D(256,(1,1),padding='SAME'))
            network.add(layers.Conv2D(512,(3,3),padding='SAME'))
        network.add(layers.Conv2D(512,(1,1),padding='SAME'))
        network.add(layers.Conv2D(1024,(3,3),padding='SAME'))
        network.add(layers.MaxPool2D((2,2),strides=(2, 2)))

        for i in range(0,2):
            network.add(layers.Conv2D(512,(1,1),padding='SAME'))
            network.add(layers.Conv2D(1024,(3,3),padding='SAME'))
        network.add(layers.Conv2D(1024,(3,3),padding='SAME'))
        network.add(layers.Conv2D(1024,(3,3),padding='SAME',strides=(2, 2)))

        network.add(layers.Conv2D(1024,(3,3),padding='SAME'))
        network.add(layers.Conv2D(1024,(3,3),padding='SAME'))

        network.add(layers.Flatten())

        network.add(layers.Dense(4096,activation="relu"))
        network.add(layers.Dense(1470,activation="sigmoid"))
        
        opt=optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        network.compile(optimizer=opt,loss=self.loss)
        return network


    def loss(self,predicts,labels):

        
        loss=list()
        
       
  
        for i in range(0,49):
            predict=predicts[30*i:30*(i+1)]
            label=labels[30*i:30*(i+1)]
           
  
            for k in range(0,2):

                
                 loss.append(tf.pow((predict[0+k*5]-label[0+k*5]),2))
                 loss.append(tf.pow((predict[1+k*5]-label[1+k*5]),2))
                 loss.append(tf.pow((tf.sqrt(predict[2+k*5])-tf.sqrt(label[2+k*5])),2))
                 loss.append(tf.pow((tf.sqrt(predict[3+k*5])-tf.sqrt(label[3+k*5])),2))
                 loss.append(tf.pow((predict[4+k*5]-label[4+k*5]),2))
                #loss_position= tf.pow((predict[0]-label[0]),2)+tf.pow((predict[1]-label[1]),2)
                #loss_size=tf.pow((tf.sqrt(predict[2])-tf.sqrt(label[2])),2)+tf.pow((tf.sqrt(predict[3])-tf.sqrt(label[3])),2)
                #loss_con=tf.pow((predict[4]-label[4]),2)


                

            for m in range(0,20):

                #loss_class=tf.pow((predict[10+m]-label[10+m]),2)
                loss.append(tf.pow((predict[10+m]-label[10+m]),2))

           
           

        
        result=tf.cast(loss,dtype='float')
        return  result









        

