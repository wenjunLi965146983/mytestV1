'''
Some tool
for DataGenerator

Edit by LWJ 2019-4-10

'''

from functools import reduce
from matplotlib import pyplot as plt
from keras import models
import numpy as np

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


def feature_map(model,image,layers_slice,images_per_row = 16):
    '''
    Input a model and a image, to show the activation of each layer
    model: input a model
    image: input a image by normalization
    layers_slice:input a list [star,end],if(end==0) end len(layers)-1

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

