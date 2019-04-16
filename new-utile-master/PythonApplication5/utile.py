'''
Some tool
for DataGenerator

Edit by LWJ 2019-4-10

'''



import os
from PIL import Image
import numpy as np
import xml
from xml.etree import ElementTree as ET
from enum import Enum
import math
import model

def get_datalist(path):
    '''
    input(str:path)
    return  a list for file in the fold
    '''
    file_list=os.listdir(path)
    return file_list  


def resize_padding(image,size,padding=True):
    '''
    input(image,size=(**,**))
    resize image without change the aspect ratio of image

    '''
    iw,ih=image.size
    w,h=size
    scale=min(w/iw,h/ih)
    nw=int(iw*scale)
    nh=int(ih*scale)
    image=image.resize((nw,nh),Image.BICUBIC)
    backGround=Image.new('RGB',size,(128,128,128))
    backGround.paste(image,((w-nw)//2,(h-nh)//2))
    return backGround


def read_image(path):
    '''
    input(str:path)
    read an image
    '''
    return Image.open(path)

def image_enhance(image,label,move=False,flip=False,rotation=False,distort=False):
    '''
    input a image and label
    return a new image and label by enhance
    '''


def read_xml(path,image_size,label_size):
    '''
    input(str:path)
    read a xml,return a normalized label
    return size,label
    [0:5] bbox1:x,y,w,h,confidence
    [5:10] bbox2:x,y,w,h,confidence
    [10,30]class
    '''
    
    label_file=open(path)
    xml_data=label_file.read()
    label_file.close() 
    label=ET.XML(xml_data)  # translate xml to string

    label_size_xml=label.find('size') #read image size
    size=np.zeros(2)
    value=label_size_xml.find('width')
    size[0]=float(value.text)
    value=label_size_xml.find('height')
    size[1]=float(value.text)

    label_object=label.findall('object')#read object

    for item in label_object:

        obj_class=object[item.find('name').text].value
        box=item.find('bndbox')
        xmin=float(box.find('xmin').text)
        ymin=float(box.find('ymin').text)
        xmax=float(box.find('xmax').text)
        ymax=float(box.find('ymax').text)

        xcenter=(xmin+xmax)/(2*size[0])
        ycenter=(ymin+ymax)/(2*size[1])
        box_width=(xmax-xmin)/size[0]
        box_height=(ymax-ymin)/size[1]

  
               
        x_cell=int(xcenter/float(1/math.sqrt(label_size[0])))
                  
        y_cell=int(ycenter/float(1/math.sqrt(label_size[0])))

        label_matrix=np.zeros(label_size)
        label_matrix[x_cell*7+y_cell,0:5]=[xcenter,ycenter,box_width,box_height,1]

        label_matrix[x_cell*7+y_cell,obj_class-5]=1

    return size,label_matrix
    








class datagenerater:
    def __init__(self,file_path,label_path,image_size=(448,448),label_size=(49,25),batch_size=32,evaluate_ratio=0.2,is_shuffle=False):
        '''
        input(file_path,label_path,image_size=(448,448),batch_size=32,evaluate_ratio=0.2)
        init datagenerater
        '''
        self.file_path=file_path
        self.label_path=label_path
        self.image_size=image_size
        self.label_size=label_size
        self.batch_size=batch_size
        self.evaluate_ratio=evaluate_ratio
        self.is_shuffle=is_shuffle
        return super().__init__()


    def get_train_data(self):
       
       '''
       return image,label in a generater
       '''

       data_list=get_datalist(self.file_path)
       number=len(data_list)
       maxnumber=int(number*(1-self.evaluate_ratio))
       index=0
       while True:
            image_batch=list()
            label_batch=list()
            for i in range(self.batch_size):
                if index==0 or index>=maxnumber:
                    index=0
                    if self.is_shuffle:
                        np.random.shuffle(data_list)
                    
                    
                image=read_image(self.file_path+data_list[index]) ##读图
                image=resize_padding(image,self.image_size)       ##填充
                image=np.array(image)
                image_batch.append(image)

                xml_name=data_list[index].split('.')[0]+".xml"
                _,label=read_xml(self.label_path+xml_name,self.image_size,self.label_size)
                label_batch.append(label)
                index+=1
            image_data=np.reshape(image_batch,(-1,self.image_size[0],self.image_size[1],3))
            label_data=np.reshape( label_batch,(-1,self.label_size[0]*self.label_size[1]))
            y_true=model.preprocess_true_boxes(label_data,(7,7),20)
            yield  image_data,y_true



class object(Enum):

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

