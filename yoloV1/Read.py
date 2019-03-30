import os
import random as rd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
    
import xml
from xml.etree import ElementTree as ET
from enum import Enum

class datagentor:
    def __init__(self, file_path,label_path,image_size=(448,448,3),batch_size=8):
        self.file_path=file_path
        self.label_path=label_path
        self.image_size=image_size
        self.batch_size=batch_size
        self.index=0
    def load_data(self):
         data_list=os.listdir(self.file_path)

         self.data_length=len(data_list)
         return data_list
   
     
    def get_batch(self):
        data_list=self.load_data()
        np.set_printoptions(threshold=10000)
 
        while True:
           
            image_batch=list()
            label_batch=list()
           
            for i in range(0,self.batch_size):
               
                label_matrix=np.zeros((49,30))
               
                image1=Image.open(self.file_path+data_list[self.index])##read image
                image=image1.resize((448,448))
                image=np.array( image)
                image=np.reshape(image,self.image_size)


                name=data_list[self.index].split('.')##read xml
                label_file=open(self.label_path+name[0]+".xml")
                xml_data=label_file.read()
                label_file.close() 
                label=ET.XML(xml_data)

                label_size=label.find('size')
                size=np.zeros(2)
                for value in label_size.iter('width'):
                    size[0]=float(value.text)
                for value in label_size.iter('height'):
                     size[1]=float(value.text)

                label_object=label.findall('object')

                for item in label_object:

                    obj_class=obj[item.find('name').text].value
                    box=item.find('bndbox')
                    xmin=float(box.find('xmin').text)
                    ymin=float(box.find('ymin').text)
                    xmax=float(box.find('xmax').text)
                    ymax=float(box.find('ymax').text)

                    xcenter=(xmin+xmax)/(2*size[0])
                    ycenter=(ymin+ymax)/(2*size[1])
                    box_width=(xmax-xmin)/size[0]
                    box_height=(ymax-ymin)/size[1]

                   # print("Xc:")
               
                    x_cell=int(xcenter/float(1/7))
                  
                    y_cell=int(ycenter/float(1/7))


                    label_matrix[x_cell*7+y_cell,0:10]=[xcenter,ycenter,box_width,box_height,1,xcenter,ycenter,box_width,box_height,1]

                    label_matrix[x_cell*7+y_cell,obj_class]=1
                  #  print(label_matrix)


               

                image_batch.append(image)
                label_batch.append(label_matrix)
                self.index+=1
                if self.index >= self.data_length :
                    self.index=0

            result1=np.reshape(image_batch,(-1,self.image_size[0],self.image_size[1],3))
            result2=np.reshape( label_batch,(-1,1470))

            yield  result1,result2

class obj(Enum):

     person=10
     bird=11
     cat=12
     cow=13
     dog=14
     horse=15
     sheep =16
     aeroplane=17
     bicycle=18
     boat=19
     bus=20
     car=21
     motorbike=22
     train =23
     bottle=24
     chair=25
     diningtable=26
     pottedplant=27
     sofa=28
     tvmonitor=29
