import sys

sys.path.append("mytest/")
sys.path.append("lwjyolo/")

#from mytest import utile
#from mytest import imagetool
#from mytest import ModelTool as MT

#from lwjyolo import yolo
#from lwjyolo import train
#from lwjyolo import convert

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
#from keras import models

'''
model 
'''


#def test():

#   #convert._main('lwjyolo/yolov3.cfg','lwjyolo/model_data/yolov3.weights','lwjyolo/model_data/yolo.h5')

#   #print('complete')

#   #anchors=train.get_anchors('lwjyolo/model_data/yolo_anchors.txt')#get yolo model
#   #model= train.create_model((416,416),anchors,2)
#   #model.save('lwjyolo/model_data/lwjyolo.h5')
#   #model.summary()
   
#   y_model=yolo.YOLO()
 

#   #gen=utile.datagenerater('E:/VOC2012/JPEGImages/','E:/VOC2012/Annotations/',(416,416),batch_size=1)
#   #data=gen.get_train_data()
#   keep_run=True
#   while keep_run:
#       image =utile.read_image('E:/VOC2013/2.jpg')
#       image=image.resize((416,416))
#       a=input('in:')
#       if a=='':
#           keep_run=False

      
            
#       #ima=Image.fromarray(image)
#       result=y_model.detect_image(image)
       
#       plt.imshow(result)
#       plt.show()
  
#def imagetest():

#    annotation_path = 'lwjyolo/model_data/yolo_annatation.txt'
#    log_dir = 'lwjyolo/logs/000/'
#    classes_path = 'lwjyolo/model_data/lemon_or_orange.txt'
#    anchors_path = 'lwjyolo/model_data/yolo_anchors.txt'
#    class_names = train.get_classes(classes_path)
#    num_classes = len(class_names)
#    anchors = train.get_anchors(anchors_path)
#    batch_size=1

#    input_shape = (416,416) # multiple of 32, hw


#    val_split = 0.1
#    with open(annotation_path) as f:
#        lines = f.readlines()
#    np.random.seed(10101)
#    np.random.shuffle(lines)
#    np.random.seed(None)
#    num_val = int(len(lines)*val_split)
#    num_train = len(lines) - num_val



#    data=train.data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes)
#    np.set_printoptions(threshold=np.NaN)
#    while True:
#         image,label=next(data)
      
#         im=np.reshape(image[0],(416,416,3))
        
         
#         label_data=np.reshape(image[1],(-1,13,13,3,7))
        
#         im*=255
     
#         pic=Image.fromarray(np.uint8(im))
         
#         for label in label_data:
#             ix=0
             
#             for xy in label:
#                 iy=0
#                 for y in xy:
#                     for l in y:
#                         if l[4]==1:
                             

#                              rect=[l[0]*416-l[2]*208,l[1]*416-l[3]*208,l[0]*416+l[2]*208,l[1]*416+l[3]*208]
#                              pic=imagetool.draw_rectangel(pic,rect)
                            
                  
                    
                       
                        
                        
#                     iy+=1
#                 ix+=1


#         label_data=np.reshape(image[2],(-1,26,26,3,7))
#         for label in label_data:
#             ix=0
             
#             for xy in label:
#                 iy=0
#                 for y in xy:
#                     for l in y:
#                         if l[4]==1:
                             

#                              rect=[l[0]*416-l[2]*208,l[1]*416-l[3]*208,l[0]*416+l[2]*208,l[1]*416+l[3]*208]
#                              pic=imagetool.draw_rectangel(pic,rect)
                            
                  
                    
                       
                        
                        
#                     iy+=1
#                 ix+=1

#         label_data=np.reshape(image[3],(-1,52,52,3,7))
#         for label in label_data:
#             ix=0
             
#             for xy in label:
#                 iy=0
#                 for y in xy:
#                     for l in y:
#                         if l[4]==1:
                             

#                              rect=[l[0]*416-l[2]*208,l[1]*416-l[3]*208,l[0]*416+l[2]*208,l[1]*416+l[3]*208]
#                              pic=imagetool.draw_rectangel(pic,rect)
                            
                  
                    
                       
                        
                        
#                     iy+=1
#                 ix+=1



          
             

#         pic.show()


       
   
  




if __name__== '__main__':
   a=np.array([[1,2]])
   b=np.array([[[2,4],[0,0],[0,0]],[[0,0],[3,6],[0,0]],[[0,0],[0,0],[4,8]]])

   print(b/a)

