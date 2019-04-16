from keras.applications import vgg16
from keras import models
from keras import layers

import keras.backend as K
import tensorflow as tf
import numpy as np
'''
Define
'''
anchors=[[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]]



'''
model 
'''


def creat_model():

    

    K.clear_session() # get a new session
    model=vgg16.VGG16(include_top=False,weights='imagenet',input_shape=(448,448,3))

    for layer in model.layers:
        layer.trainable=False

    x=model.output
    x=layers.MaxPool2D((2,2))(x)
    x=layers.Flatten()(x)
    x=layers.Dense(4096,activation='relu')(x)
    x=layers.Dense(7*7*3*25,activation='relu')(x)
    y=layers.Reshape((7,7,3,25))(x)

    model_body=models.Model(model.input,y)
    y_true=layers.Input((7,7,3,25))
    
    model_loss=layers.Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': 20, 'ignore_threshold': 0.5})([y_true,model_body.output])
    
    model = models.Model([model_body.input, y_true], model_loss)

    model.summary()
    return model




'''
layers
'''


def yolo_head(features,anchors,num_classes,input_shape, calc_loss=False):
    '''
    解析YOLO OUTput
    Convert final layer features to bounding box parameters.

    Parameters
    ----------
    features:shape=[batch, height, width, num_anchors*box_params]
    '''
    num_anchors = len(anchors)# shape=(3,2)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])##create anchors tensor


    grid_shape=K.shape(features)[1:3]  ##list,[x,y]

    grid_y=K.tile(K.reshape(K.arange(0,stop=grid_shape[0]),[-1,1,1,1]),[1, grid_shape[1], 1, 1])

    grid_x=K.tile(K.reshape(K.arange(0,stop=grid_shape[1]),[1,-1,1,1]),[grid_shape[0],1,1,1])

    grid =K.concatenate([grid_x,grid_y])## 2dim list 


    grid=K.cast(grid,K.dtype(features))

    features=K.reshape(features,[-1,grid_shape[0],grid_shape[1],3,num_classes+5])

    box_xy=(K.sigmoid(features[...,:2])+grid)/K.cast(grid_shape[::-1],K.dtype(features))

    box_wh=K.exp(features[...,2:4])*anchors_tensor/K.cast(input_shape[::-1],K.dtype(features))

    box_confidence=K.sigmoid(features[..., 4:5])
    box_class_probs = K.sigmoid(features[..., 5:])

    if calc_loss == True:
        return grid, features, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs



def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou



def yolo_loss(args,anchors, num_classes,ignore_threshold=0.5,print_loss=False):
    '''
    Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs:list of tensor,the output of yolo_body or tiny_yolo_body
    y_true:list of array,the output of preprocess_true_boxes
    num_classes:integer
    ignore_threshold:float,the iou threshold whether to ignore object confidence loss

    Return
    ------
    loss:tensor,shape(none,49*3*(num_classes+5))

    '''

    loss=0
    num_layers = len(anchors)//3
    y_true=args[0]
    y_pre=args[1]
    
    #_y_true=np.reshape(y_true,(-1,7,7,3,num_classes+5))  ##规范Tensor
    #_y_pre=np.reshape(y_pre,(-1,7,7,3,num_classes+5))
    input_shape=K.cast(K.shape(y_true)[1:3]*64,K.dtype(y_true[0]))
    grid_shapes=K.cast(K.shape(y_true)[1:3],K.dtype(y_true[0]))


    object_mask=y_true[...,4:5]   
    true_classes_pros=y_true[...,5:]

    
    m = K.shape(y_pre)[0] # batch size, tensor
   
    mf = K.cast(m, K.dtype(y_pre))



    grid, raw_pred, pred_xy, pred_wh = yolo_head(y_pre,anchors[6:9], num_classes, input_shape, calc_loss=True)## convert the output of yolo_model

    pre_box=K.concatenate([pred_xy,pred_wh])


    #create y_true tensor for calculate loss
    raw_true_xy=y_true[...,0:2]*grid_shapes[::-1]-grid
    raw_true_wh=K.log(y_true[...,2:4]/anchors[6:9]*input_shape[::-1])
    raw_true_wh=K.switch(object_mask,raw_true_wh,K.zeros_like(raw_true_wh))
    box_loss_scale=2-y_true[...,2:3]*y_true[...,3:4]#size
    true_class_probs=y_true[...,5:]
    #Find iagnore mask,iterate over each batch
    ignore_mask=tf.TensorArray(K.dtype(y_true[0]),size=1,dynamic_size=True)
    object_mask_bool = K.cast(object_mask, 'bool')
    def loop_body(b,ignore_mask):
        true_box=tf.boolean_mask(y_true[b,...,0:4],object_mask_bool[b,...,0])# set mask for x,y,w,h
        iou = box_iou(pre_box[b], true_box)         #calculate iou
        best_iou = K.max(iou, axis=-1)
        ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_threshold, K.dtype(true_box)))
        return b+1, ignore_mask
    _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])##ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m,   loop_body,    [0, ignore_mask])
    ignore_mask = ignore_mask.stack()
    ignore_mask = K.expand_dims(ignore_mask, -1)
    # K.binary_crossentropy is helpful to avoid exp overflow

    xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
    wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
    confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
    class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

    xy_loss = K.sum(xy_loss) / mf
    wh_loss = K.sum(wh_loss) / mf
    confidence_loss = K.sum(confidence_loss) / mf
    class_loss = K.sum(class_loss) / mf
    loss += xy_loss + wh_loss + confidence_loss + class_loss
    if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss


def preprocess_true_boxes(true_boxes,input_shape,num_classes,anchors=anchors):
    '''
    Preprocess true boxes to training input format

    Parameters
    ----------
    true boxes :array,shape(m,T,5)
    '''
    num_layers = len(anchors)//3

    true_boxes = np.array(true_boxes, dtype='float32')

    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = true_boxes[..., 0:2] 
    boxes_wh = true_boxes[..., 2:4] 
    m = true_boxes.shape[0]

    grid_shapes=[7,7]

    y_true =np.zeros((m,grid_shapes[0],grid_shapes[1],num_layers,5+num_classes))

    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)


        for t, n in enumerate(best_anchor):
           
                if n in anchors:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[0]).astype('int32')
                    k = anchor_mask.index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[b, j, i, k, 4] = 1
                    y_true[b, j, i, k, 5+c] = 1

    return y_true


if __name__ == '__main__':
    creat_model()
