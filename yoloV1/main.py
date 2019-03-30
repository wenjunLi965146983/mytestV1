from Read import *
from model import *
import keras

def main():
    train=datagentor('E:/VOC2012/JPEGImages/','E:/VOC2012/Annotations/')
    data=train.get_batch()

    mo=model((448,448,3))
    network=mo.get_model()
    network.summary()
    
    network.fit_generator(data,steps_per_epoch=100,epochs=30)
 
main()
