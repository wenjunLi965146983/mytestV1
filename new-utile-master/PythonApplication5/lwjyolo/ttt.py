from keras import backend as K



a=K.tile(K.reshape(K.arange(0, stop=10), [-1, 1, 1, 1]),
        [1, 13, 1, 1])
b=K.tile(K.reshape(K.arange(0, stop=13), [1, -1, 1, 1]),
        [10, 1, 1, 1])
c=K.concatenate([a,b])
print(K.int_shape(c))
sess=K.get_session()
print(c.eval(session=sess))



