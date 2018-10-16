import tensorflow as tf
import numpy as np
import reader


import model
from tensorflow.contrib.factorization.examples.mnist import fill_feed_dict
import os 
import matplotlib.pyplot as plt






logs_train_dir=r'E:\eclips\YOLO_tensorFlow\save_net/'

def train():
     # 数据集
     print("start")
     image_dir = r'E:\VOC2013\JPEGImages/'   #My dir--20170727-csq  
     xml_dir=r'E:\VOC2013\Annotations/'
    
    
    
    
    #获取图片和参数
     imagesname,labels,w_h_s,number=reader.input_data()
     label_data=reader.data_normalizer(labels,w_h_s,number)
     label_32=tf.cast(label_data,tf.float32)
     print( label_32)
     print("start2")
     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
     image_batch=[]
     image_batch,label_batch=reader.get_batch(imagesname, label_data,number)
     print(image_batch)
    
     

     
     #tensor=np.array(image_batch)
     sess=tf.Session()
    
    
     
    
     coord = tf.train.Coordinator()
     ps1,ps2,ps3=model.model(image_batch,True)
     
     scale1, scale2, scale3 = model.scales( ps1,ps2,ps3,True )
     loss=model.loss(scale1,label_batch)
     tf.squeeze(loss,2)
     print(loss)
     tf.summary.scalar('loss',loss)
     train_op = model.op(loss, 0.01)
     summary_op = tf.summary.merge_all()  
     train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)  
     saver = tf.train.Saver()
    
     with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer()) 
        q=tf.train.start_queue_runners(sess=sess, coord=coord)
         #sess.run(tensor)
        for step in range(10000):
            
            #sess.run([ps1,ps2,ps3])
            #sess.run([scale1, scale2, scale3])
           
            op,loss_result=sess.run([train_op,loss])

            if step % 50 == 0:  
                print(step) 
                print(loss_result)
                summary_str = sess.run(summary_op)  
                train_writer.add_summary(summary_str , step)  

            if step % 2000 == 0 or (step + 1) == 10000:  
                # 每隔2000步保存一下模型，模型保存在 checkpoint_path 中
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')  
                saver.save(sess, checkpoint_path, global_step=step)   
       
     print('finish')
    
    
   

    
     

print("get batch") 
train()
   

        
"""
     labels=tf.placeholder(tf.int8,[448,448],name='labels')
     summary_op = tf.summary.merge_all()  
     sess=tf.Session()
     train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)  
     saver = tf.train.Saver()  
     sess.run(tf.global_variables_initializer())  
"""    
    