import model
from keras import models
import utile as lwj
from keras.optimizers import Adam
def train():


    generater=lwj.datagenerater("I:/VOC2012/JPEGImages/","I:/VOC2012/Annotations/",batch_size=32)
    datagenerater=generater.get_train_data()
    network=model.creat_model()

    network.compile(optimizer=Adam(lr=1e-3),loss={'yolo_loss': lambda y_true, y_pred: y_pred} )
    network.fit_generator(datagenerater,steps_per_epoch=535,epochs=10)
    network.save('yoloV2_1.h5')


if __name__ =='__main__':
    train()