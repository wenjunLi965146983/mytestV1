import tensorflow as tf
from tensorflow.python.ops.nn_ops import leaky_relu
from scipy.constants.constants import alpha

def Leaky_Relu( input, alpha = 0.01 ):
    output = tf.maximum( input, tf.multiply( input, alpha ) )

    return output


def conv2d( inputs, filters, shape, stride = ( 1, 1 ), training = True ):
    layer = tf.layers.conv2d( inputs,
                              filters,
                              shape,
                              stride,
                              padding = 'SAME',
                              kernel_initializer=tf.truncated_normal_initializer( stddev=0.01 ) )

    layer = tf.layers.batch_normalization( layer, training = training )

    layer = Leaky_Relu( layer )

    return layer


def Res_conv2d( inputs, shortcut, filters, shape, stride = ( 1, 1 ), training = True ):
    conv = conv2d( inputs, filters, shape, training = training )
    Res = Leaky_Relu( conv + shortcut )

    return Res


def model( inputs, training ):
    layer = conv2d( inputs, 32, [3, 3], training = training )
    layer = conv2d( layer, 64, [3, 3], ( 2, 2 ), training = training )
    shortcut = layer

    layer = conv2d( layer, 32, [1, 1], training = training )
    layer = Res_conv2d( layer, shortcut, 64, [3, 3], training = training )

    layer = conv2d( layer, 128, [3, 3], ( 2, 2 ), training = training )
    shortcut = layer

    for _ in range( 2 ):
        layer = conv2d( layer, 64, [1, 1], training = training )
        layer = Res_conv2d( layer, shortcut, 128, [3, 3], training = training )

    layer = conv2d( layer, 256, [3, 3], ( 2, 2 ), training = training )
    shortcut = layer

    for _ in range( 8 ):
        layer = conv2d( layer, 128, [1, 1], training = training )
        layer = Res_conv2d( layer, shortcut, 256, [3, 3], training = training )
    pre_scale3 = layer

    layer = conv2d( layer, 512, [3, 3], ( 2, 2 ), training = training )
    shortcut = layer

    for _ in range( 8 ):
        layer = conv2d( layer, 256, [1, 1], training = training )
        layer = Res_conv2d( layer, shortcut, 512, [3, 3], training = training )
    pre_scale2 = layer

    layer = conv2d( layer, 1024, [3, 3], ( 2, 2 ), training = training )
    shortcut = layer

    for _ in range( 4 ):
        layer = conv2d( layer, 512, [1, 1], training = training )
        layer = Res_conv2d( layer, shortcut, 1024, [3, 3], training = training )
    pre_scale1 = layer

    return pre_scale1, pre_scale2, pre_scale3
    
def get_layer2x( layer_final,pre_scale):
    layer2x = tf.image.resize_images(layer_final,
                                     [2 * tf.shape(layer_final)[1], 2 * tf.shape(layer_final)[2]])
    layer2x_add = tf.concat( [layer2x, pre_scale], 3 )

    return layer2x_add
                   
def scales( layer, pre_scale2, pre_scale3, training ):
    layer_copy = layer
    layer = conv2d( layer, 512, [1, 1], training = training )
    layer = conv2d( layer, 1024, [3, 3], training = training )
    layer = conv2d(layer, 512, [1, 1], training = training )
    layer_final = layer
    layer = conv2d(layer, 1024, [3, 3], training = training )

    '''--------scale_1--------'''
    scale_1 = conv2d( layer, 255, [1, 1], training = training )

    '''--------scale_2--------'''
    layer = conv2d( layer_final, 256, [1, 1], training = training )
    layer = get_layer2x( layer, pre_scale2 )

    layer = conv2d( layer, 256, [1, 1], training = training )
    layer= conv2d( layer, 512, [3, 3], training = training )
    layer = conv2d( layer, 256, [1, 1], training = training )
    layer = conv2d( layer, 512, [3, 3], training = training )
    layer = conv2d( layer, 256, [1, 1], training = training )
    layer_final = layer
    layer = conv2d( layer, 512, [3, 3], training = training )
    scale_2 = conv2d( layer, 255, [1, 1], training = training )

    '''--------scale_3--------'''
    layer = conv2d( layer_final, 128, [1, 1], training = training )
    layer = get_layer2x( layer, pre_scale3 )

    for _ in range( 3 ):
        layer = conv2d( layer, 128, [1, 1], training = training )
        layer = conv2d( layer, 256, [3, 3], training = training )
    scale_3 = conv2d( layer,75, [1, 1], training = training )

    scale_1 = tf.abs( scale_1 )
    scale_2 = tf.abs( scale_2 )
    scale_3 = tf.abs( scale_3 )

    return scale_1, scale_2, scale_3
    

    
    
    
  









def loss(batch_inputs,batch_labels):
    batch_loss=[]
    count=0
    loss_sigle=0
    for image_num in range(batch_inputs.shape[0]):
        for y in range(batch_inputs.shape[1]):
            predict_class=[]
            label_class=[]
            for x in  range(batch_inputs.shape[2]):
                loss_sum=0
                t=batch_inputs[image_num][y][x]
                t1=batch_labels[image_num][y][x]
                print(x)
                for i in range(3):
                    #predict_x=batch_inputs[image_num][y][x][i*25]
                    #predict_y=batch_inputs[image_num][y][x][i*25+1]
                    #predict_width=batch_inputs[image_num][y][x][i*25+2]
                    #predict_height=batch_inputs[image_num][y][x][i*25+3]
                    #predict_objectness=batch_inputs[image_num][y][x][i*25+4]
                   
                  
                    

                    #label_x=batch_labels[image_num][y][x][i*25]
                    #label_y=batch_labels[image_num][y][x][i*25+1]
                    #label_width=batch_labels[image_num][y][x][i*25+2]
                    #label_height=batch_labels[image_num][y][x][i*25+3]
                    #label_objectness=batch_labels[image_num][y][x][i*25+4]
                    #pretect_class = batch_inputs[image_num][y][x][i * 25 + 5 : i * 25 + 5 + 20]
                    #label_class = batch_labels[image_num][y][x][i * 25 + 5 : i * 25 + 5 + 20]
                    
                   

                    predict_x=t[i*25]
                    predict_y=t[i*25+1]
                    predict_width=t[i*25+2]
                    predict_height=t[i*25+3]
                    predict_objectness=t[i*25+4]
                   
                  
                    

                    label_x=t1[i*25]
                    label_y=t1[i*25+1]
                    label_width=t1[i*25+2]
                    label_height=t1[i*25+3]
                    label_objectness=t1[i*25+4]
                    pretect_class = t1[i * 25 + 5 : i * 25 + 5 + 20]
                    label_class = t1[i * 25 + 5 : i * 25 + 5 + 20]

                    loss_pos=position_loss(predict_x,predict_y,predict_width,predict_height,label_x,label_y,label_width,label_height)
                    IOU=get_IOU(predict_x,predict_y,predict_width,predict_height,label_x,label_y,label_width,label_height)
                    loss_obj=objectness_loss(IOU,predict_objectness,label_objectness)
                    loss_class=class_loss(pretect_class,label_class)

                    loss_sum+=loss_pos+loss_obj+ loss_class
                
            loss_sigle+=loss_sum
        batch_loss.append(loss_sigle)
        print( batch_loss)
       

    batch_loss=tf.cast(batch_loss,dtype=tf.float32)
  
    return batch_loss

def op(loss, learning_rate):  
    with tf.name_scope('optimizer'):  
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)  
        global_step = tf.Variable(0, name='global_step', trainable=False)  
        train_op = optimizer.minimize(loss, global_step= global_step)  
    return train_op  


         
            
            
        
       
        
                  
               




def objectness_loss(input, switch, l_switch, alpha = 0.5):
    
    IOU_loss = tf.square( l_switch - input * switch )
    loss_max = tf.square( l_switch * 0.5 - input * switch )
    
    IOU_loss = tf.cond( IOU_loss < loss_max, lambda : tf.cast( 1e-8, tf.float32 ), lambda : IOU_loss )

    IOU_loss = tf.cond( l_switch < 1, lambda : IOU_loss * alpha, lambda : IOU_loss )

    return IOU_loss
    
def position_loss(p_x,p_y,p_w,p_h,l_x,l_y,l_w,l_h,alpha=5):
    
    
    P_loss=tf.square(p_x-l_x)+tf.square(p_y-l_y)*alpha
    s_loss=tf.square(tf.sqrt(p_w)-tf.sqrt(l_w))+tf.square(tf.sqrt(p_h)-tf.sqrt(l_h))
    
    return P_loss+s_loss
    
def class_loss(inputs,labels):
   c_l=tf.square(inputs-labels)
 

   return tf.reduce_sum(c_l)
        
def get_IOU(p_x,p_y,p_w,p_h,l_x,l_y,l_w,l_h):
    x_min=tf.maximum(p_x,l_x)
    y_min=tf.maximum(p_y,l_y)

    p_x_max=p_x+p_w
    p_y_max=p_y+p_h
    l_x_max=l_x+l_w
    l_y_max=l_y+l_h
    x_max=tf.minimum(p_x_max,l_x_max)
    y_max=tf.minimum(p_y_max,l_y_max)

    area=(x_max-x_min)*(y_max-y_min)

    all_area=tf.cond((p_w*p_h+l_w*l_h-area)<=0,lambda:tf.cast( 1e-8, tf.float32 ),lambda:(p_w*p_h+l_w*l_h-area))
    IOU = area / all_area
    return IOU






    
def calculate_max(x,w):
    return x+w/2   
    
def calculate_min(x,w):
    return x-w/2    





def op(loss,learning_rate):
      
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)  
    global_step = tf.Variable(0, name='global_step', trainable=False)  
    train_op = optimizer.minimize(loss, global_step= global_step)  
    return train_op 
    