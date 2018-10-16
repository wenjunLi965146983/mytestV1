import xml.dom.minidom as xdm
import os 
import numpy as np
import tensorflow as tf

def xml_reader(file):
    
    DOMTree=xdm.parse(r"E:\VOC2013\Annotations\{0}.xml".format(file))
    DOMNode=DOMTree.documentElement
    fileName=DOMNode.getElementsByTagName("filename")[0].childNodes[0].data
   
    objects=DOMNode.getElementsByTagName("object")
    size=DOMNode.getElementsByTagName("size")[0]
    w=size.getElementsByTagName("width")[0].childNodes[0].data
    h=size.getElementsByTagName("height")[0].childNodes[0].data

    w_h=(w,h)
    
    datas=[]
    for obj in objects:
         name=obj.getElementsByTagName("name")[0]
       
         xmin=obj.getElementsByTagName("xmin")[0]
         ymin=obj.getElementsByTagName("ymin")[0]
         xmax=obj.getElementsByTagName("xmax")[0]
         ymax=obj.getElementsByTagName("ymax")[0]
        
         data=(name.childNodes[0].data,
               xmin.childNodes[0].data,
               ymin.childNodes[0].data,
               xmax.childNodes[0].data,
               ymax.childNodes[0].data)
         
         datas.append(data)
    return fileName,w_h,datas 
    
def input_data():
    
    images=[]
    labels=[]
    w_h_s=[]
   
    file_dir = r'E:\VOC2013\JPEGImages/' 
    for file in os.listdir(file_dir):
       
        images.append(list(file.split("*")))
      
    np.random.shuffle(images)
    number=images.__len__()
    for i in range(images.__len__()):
 
        name=images[i]
       
        str=''.join(name)
      
        filename,w_h,data=xml_reader(str.split('.')[0])
        labels.append(data)
        w_h_s.append(w_h)
    return images,labels,w_h_s,number

def get_batch(image,label,number):
    # 转换数据为 ts 能识别的格式
    image = tf.cast(image,tf.string)
    image_batch=[]
    label_batch=[]
 
    print(image[0][0])
    print(number)
   
   

    input_q=tf.train.slice_input_producer([image,label])
    image_contents = tf.read_file(image[0][0])
  
    label =   input_q[1]
  
  
    file_dir = r'E:\VOC2013\JPEGImages/' 
    name=file_dir+input_q[0][0]
    print(name)
    image_contents = tf.read_file( name)
    im = tf.image.decode_jpeg(image_contents,channels =3)
    im = tf.image.resize_image_with_crop_or_pad(im, 256, 256)
    im = tf.image.per_image_standardization(im)

    image_batch, label_batch =tf.train.batch([im,label],batch_size = 2, num_threads = 64, capacity = 256)
   
        
    label_batch = tf.reshape(label_batch , [2,8,8,75])
    # 转化图片
    image_batch = tf.cast(image_batch,tf.float32)
    label_batch  = tf.cast(label_batch,tf.float32)
    return  image_batch, label_batch






def data_normalizer(labels,w_h_s,number):
    
    class_map={
        'person':5,
        'bird':6,
        'cat':7,
        'cow':8,
        'dog':9,
        'horse':10,
        'sheep':11,
        'aeroplane':12,
        'bicycle':13,
        'boat':14,
        'bus':15,
        'car':16,
        'motorbike':17,
        'train':18,
        'bottle':19,
        'chair':20,
        'diningtable':21,
        'potteplant':22,
        'sofa':23,
        'tvmonitor':24,
        'pottedplant':25 
        }
    
    
    height_width=[]
    label_batch=np.add( np.zeros( [number,8, 8,75] ), 1e-8 )
    i=0
    for label in labels:
        
        #读取图片size
       
        size=w_h_s[i]
        w=int(size[0])
        h=int(size[1])
        ++i
      
        #产生label模版
        label_temple = np.add( np.zeros( [8, 8, 75] ), 1e-8 )


          #读取图片object
          #图片缩放比例：32
        scale=32
      
        for element in label:
            
             try:
              
                #object参数
                xmin=float(element[1])
                ymin=float(element[2])
                xmax=float(element[3])
                ymax=float(element[4])
                
                #计算中心
                x=((xmax+xmin)/w)*128
                y=((ymax+ymin)/h)*128
                
                #计数box位置
                obj_c_x=int(x/32)
                obj_c_y=int(y/32)

                if obj_c_x>=8:
                  obj_c_x=7
                if obj_c_y>=8:
                  obj_c_y=7

                obj_w=((xmax-xmin)/w)*256
                obj_h=((ymax-ymin)/w)*256

                class_label = class_map[element[0]]

                for i in range(3):
                    label_temple[int(obj_c_y),int(obj_c_x),i*25]=x
                    label_temple[int(obj_c_y),int(obj_c_x),i*25+1]=y
             
                    label_temple[int(obj_c_y),int(obj_c_x),i*25+2]= obj_w
                    label_temple[int(obj_c_y),int(obj_c_x),i*25+3]= obj_h
                    label_temple[int(obj_c_y),int(obj_c_x),i*25+4]=1
                    label_temple[int(obj_c_y),int(obj_c_x),i*25+int(class_label)]=0.9

              
               
             except TypeError:
                
                 print(element[0])
           
   
        label_batch[i]=label_temple
        result=tf.convert_to_tensor(label_batch)
     
      
     
    return result

     
        
      