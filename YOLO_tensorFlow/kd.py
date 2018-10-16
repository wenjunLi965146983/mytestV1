import tensorflow as tf 
import model
import os 
import numpy as np
 

def reader():
     file_dir = r'E:\VOC2013\JPEGImages/2007_000027.jpg' 
     image_contents = tf.read_file(file_dir)
     im = tf.image.decode_jpeg(image_contents,channels =3)
     im = tf.image.resize_image_with_crop_or_pad(im, 256, 256)
     im = tf.image.per_image_standardization(im)
     im=tf.reshape(im,[1,256,256,3])
   
     return(im)
def evaluate_one_image():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
       
    im=reader()
    print(im)
    sess=tf.Session()
    s_im=sess.run(im)
    logs_train_dir=r'E:\eclips\YOLO_tensorFlow\save_net/'
    image=tf.placeholder(tf.float32,[1,256,256,3],name='image')
    ps1,ps2,ps3=model.model(image,True)
    
    scale1, scale2, scale3 = model.scales( ps1,ps2,ps3,True )
    saver = tf.train.Saver() 
    with tf.Session() as sess: 
         print('Loading') 
         ckpt = tf.train.get_checkpoint_state(logs_train_dir)  
         if ckpt and ckpt.model_checkpoint_path:  
             global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]  
             saver.restore(sess, ckpt.model_checkpoint_path)  
             print('OK: %s' % global_step)  
         else:  
             print('fail')
         np.set_printoptions(threshold=10000000)
         prediction = sess.run(scale1, feed_dict={image: s_im})
         print(prediction)
              
      
                
       
evaluate_one_image()



