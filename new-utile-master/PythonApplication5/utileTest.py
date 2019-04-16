import utile as lwj
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
def main():


   generater=lwj.datagenerater("I:/VOC2012/JPEGImages/","I:/VOC2012/Annotations/",batch_size=32)
   datagenerater=generater.get_train_data()
   keep_run=True
   while keep_run:
       a=input('intput something')
       if a=="":
           keep_run=False

       traindata=next(datagenerater)
       image=traindata[0]
      
       print(np.shape(traindata[1]))



if __name__ == "__main__":
    main()

