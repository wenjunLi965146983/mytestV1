'''
Some tool
for DataGenerator

Edit by LWJ 2019-4-10

'''

from functools import reduce
from matplotlib import pyplot as plt
from keras import models
import numpy as np
from keras.models import Model
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras import backend as K
def compose(*funcs):
    """
    Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x:reduce(lambda v,f:f(v),funcs,x)
    # return a new function:f(V(x))
    if funcs: 
        return reduce(lambda f,g:lambda *a,**kw:g(f(*a,**kw)),funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')



##yolo model


def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( DarknetConv2D(*args, **no_bias_kwargs), BatchNormalization(),LeakyReLU(alpha=0.1))

def res(x,num_filters, num_blocks):

     '''A series of resblocks starting with a downsampling Convolution2D'''
     # Darknet uses left and top padding instead of 'same' mode
     x = ZeroPadding2D(((1,0),(1,0)))(x)

     x=DarknetConv2D_BN_Leaky(num_filters,(3,3),strides=(2,2))(x)
     for i in range(num_blocks):
         y=compose(
             DarknetConv2D_BN_Leaky(num_filters//2,(1,1)),
             DarknetConv2D_BN_Leaky(num_filters,(3,3)))(x)
         x=Add()([x,y])

     return x

def Darknet53(x):
      '''Darknent body having 52 Convolution2D layers'''
      x=DarknetConv2D_BN_Leaky(32,(3,3))(x)
      x=res(x,64,1)
      x=res(x,128,2)
      x=res(x,256,8)
      x=res(x,512,8)
      x=res(x,1024,4)
      return x

def last_layer(x, num_filters, out_filters):

       '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
       x=compose(DarknetConv2D(num_filters,(1,1)),
                 DarknetConv2D(num_filters*2,(3,3)),
                 DarknetConv2D(num_filters,(1,1)),
                 DarknetConv2D(num_filters*2,(3,3)),
                 DarknetConv2D(num_filters,(1,1))
                 )(x)
       y=compose(DarknetConv2D(num_filters,(3,3)),
                 DarknetConv2D(out_filters,(1,1)))(x)
           
       return x,y

def yolo_model(inputs, num_anchors, num_classes):
     """Create YOLO_V3 model CNN body in Keras."""
     darknet = Model(inputs, Darknet53(inputs))

     ##输出y1 (13,13)
     x,y1=last_layer(darknet.output,512, num_anchors*(num_classes+5))

     ##输出y2 (26,26)
     x=compose(DarknetConv2D_BN_Leaky(256,(1,1)),
               UpSampling2D(2))(x)
     x=Concatenate()([x,darknet.layers[152].output])
     x,y2=last_layer(x,256, num_anchors*(num_classes+5))


     ##输出y3 (52,52)
     x=compose(DarknetConv2D_BN_Leaky(256,(1,1)),
               UpSampling2D(2))(x)
     x=Concatenate()([x,darknet.layers[92].output])
     x,y3=last_layer(x,128,num_anchors*(num_classes+5))

     return Model(inputs,[y1,y2,y3])

def yolo_head(yolo_body_output,anchors,num_classes,input_shape,calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)# shape=(3,2)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
    # Reshape to batch, height, width, num_anchors, box_params.
    output_shape=K.shape(yolo_body_output)
    grid_shape=output_shape[1:3]


    grid_x=K.tile(K.reshape(K.arange(0,stop=grid_shape[0]),[-1,1,1,1]),[1,grid_shape[1],1,1])#shape[grid_shape[0],grid_shape[1],1,1]
    grid_y=K.tile(K.reshape(K.arange(0,stop=grid_shape[1]),[1,-1,1,1]),[grid_shape[0],1,1,1])#shape[grid_shape[0],grid_shape[1],1,1]

    grid=K.concatenate([grid_x,grid_y])
    grid=K.cast(grid,K.dtype(yolo_body_output))##shape[grid_shape[0],grid_shape[1],1,2]

    yolo_body_output = K.reshape(
        yolo_body_output, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])#shape=(batch,w,h,3,25)

    box_xy=(K.sigmoid(yolo_body_output[...,:2])+grid)/K.cast(grid_shape,K.dtype(yolo_body_output))              ##shape=batch,w,h,3,2
    box_wh=(K.exp(yolo_body_output[...,2:4])*anchors_tensor)/K.cast(input_shape,K.dtype(yolo_body_output))

    box_confidence=K.sigmoid(yolo_body_output[...,4:5])
    box_class_probabilities=K.sigmoid(yolo_body_output[...,5:])

    if calc_loss == True:
        return grid, yolo_body_output, box_xy, box_wh
    return   box_xy, box_wh, box_confidence,box_class_probabilities

def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)for batch
    Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[...,4]<num_classes).all(),'label 的class_id必须小于num_classes'

    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')

    true_wh=true_boxes[...,2:4]-true_boxes[...,0:2]##box 的w，h 像素
    true_boxes[...,2:4]=true_wh/input_shape ##相对的w,h
    true_xy_center=(true_boxes[...,0:2]+true_boxes[...,2:4])/2
    true_boxes[...,0:2]=true_xy_center/input_shape

    grid_shape=[input_shape//{0:32,1:16,2:8}[l] for l in range(num_layers)]

    y_true=[np.zeros(
                      (grid[l][0],grid[l][1],len(anchor_mask[l]),num_classes+5), 
                      dtype='float32') 
            for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = true_wh[..., 0]>0


    m = true_boxes.shape[0]

    for i in range(m):

       # Discard zero rows.
       wh=true_wh[i,valid_mask[i]]
       if valid_mask[i]:continue
       wh=np.expand_dims(wh,-2)

       box_maxes = wh / 2.
       box_mins = -box_maxes

       intersect_mins=np.minimum(anchor_maxes,box_maxes)
       intersect_maxs=np.maximum(anchor_mins,box_mins)
       intersect_whs = np.maximum(intersect_maxes - intersect_mins, 0.)
       intersect_aeras= intersect_whs[...,0]* intersect_whs[...,1]

       box_aeras=wh[...,0]*wh[...,1]
       anchor_aeras=anchors[...,0]*anchors[...,1]

       iou=intersect_aeras/(box_aeras+anchor_aeras-intersect_aeras)

       ##find best iou
       best_anchor=np.argmax(iou,axis=-1)
       for t,n in enumerate(best_anchor):

           for l in range(num_layers):
                if n in anchor_mask[l]:

                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][0]).astype('int32')  ##对应当前layers 第n个grid w
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][1]).astype('int32')  ##对应当前layers 第n个grid h
                    k = anchor_mask[l].index(n)  ##对应当前layers 的box number
                    c = true_boxes[b,t, 4].astype('int32') ##class
                    y_true[l][b, i, j, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, i, j, k, 4] = 1
                    y_true[l][b, i, j, k, 5+c] = 1

       return y_true
 

  



def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(..., 4), xywh
    b2: tensor, shape=(..., 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2) #shape=(N,1,4)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0) #shape=(1,9,4)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)##shape=(N,9,4)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)##shape=(N,9,4)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]##(N,9,1)
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]##(N,9,1)
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]##(N,9,1)
    iou = intersect_area / (b1_area + b2_area - intersect_area)##(N,9,1)

    return iou##(N,9,1)



def yolo_loss(layer_out,anchors,num_classes,ignore_thresh=.5, print_loss=False):

     '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''

     #assert len(layer_out)<1,'loss function input 为零'
     num_layer=len(anchors)//3

     y_pre=layer_out[0:num_layer]
     y_true=layer_out[num_layer:]

     anchor_mask=[[6,7,8],[3,4,5],[0,1,2]] if num_layer==3 else [[4,5,6],[1,2,3]]


     input_shape=K.cast(K.shape(y_pre[0])[1:3],K.dtype(y_true[0]))


     loss=0
     for i in range(num_layer):
         grid, raw_pred, pred_xy, pred_wh = yolo_head(y_pre[i],
             anchors[anchor_mask[i]], num_classes, input_shape, calc_loss=True)

         pred_box=K.concatenate(pred_xy,pred_wh)

         # Darknet raw box to calculate loss.
         object_mask = y_true[l][..., 4:5]
         true_class_probs = y_true[l][..., 5:]

         raw_true_xy = y_true[l][..., :2]*grid_shapes[l] - grid ##局部  y_true[l].shape=(N,W,H,3,N+5)  shape=(N,w,h,2)
         raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape)## 相对anchors 大小  shape=(N,w,h,2)
         raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
         box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]


        
     return loss
     



## creat yolo model
def feature_map(model,image,layers_slice,images_per_row = 16):
    '''
    Input a model and a image, to show the activation of each layer
    model: input a model
    image: input a image by normalization
    layers_slice:input a list [star,end],if(end==0) end len(layers)-1
    if result  dimension is not bigger than 3,while be array,and save in feature_map.txt
    '''
   

  

    def get_layer(number):
        result=number if number>0 else len(model.layers)+number
        return result




    
    layer_outputs = [layer.output for layer in model.layers[get_layer(layers_slice[0]):get_layer(layers_slice[1])]] 
    
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(image) 


    layer_names = []
    
    for layer in layer_outputs:     
        layer_names.append(layer.name) #层的名称，这样你可以将这些名称画到图中
 
     
 
    for layer_name, layer_activation in zip(layer_names, activations):     #显示特征图  

        if len(layer_activation.shape)>2:
           n_features = layer_activation.shape[-1]   #特征图中的特征个数
 
           size = layer_activation.shape[1]   #特征图的形状为 (1, size, size, n_features)
 
           n_cols = n_features // images_per_row    #在这个矩阵中将激活通道平铺
           
           display_grid = np.zeros((size * n_cols, images_per_row * size)) 
 
           for col in range(n_cols):           #将每个过滤器平铺到 一个大的水平网格中
               for row in range(images_per_row):             
                   channel_image = layer_activation[...,col * images_per_row + row]             
                   channel_image -= channel_image.mean()           #对特征进行后 处理，使其看 起来更美观    
                   channel_image /= channel_image.std()             
                   channel_image *= 64             
                   channel_image += 128             
                   channel_image = np.clip(channel_image, 0, 255).astype('uint8')             
                   display_grid[col * size : (col + 1) * size,                            
                                row * size : (row + 1) * size] = channel_image 
 

           scale = 1. / size     
           plt.figure(figsize=(scale * display_grid.shape[1],                         
                        scale * display_grid.shape[0]))     
           plt.title(layer_name)     
           plt.grid(False)     
           plt.imshow(display_grid, aspect='auto', cmap='viridis')
           plt.show()
        else:
            result =layer_activation
            print(layer_name)
            np.savetxt('feature_map',result)

