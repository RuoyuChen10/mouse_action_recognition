import numpy as np
import cv2
import os

class Load:
    '''
    This class using for .....
    '''
    def __init__(self):
        self.datasets_txt = './dataset.txt'
        self.img_size = (224,224)
    
    def read_folder_datasets(txt_serial=self.datasets_txt):
    '''
    给一行，读一个标签和动作序列
    '''
        y = int(txt_serial.split(' ')[-1])
        x = []
        for i in txt_serial.split(' ')[:-1]:
            img = cv2.imread(i)
            img = cv2.resize(img,self.img_size)
            if img is not None:
                x.append(img)
        if len(x)==len(txt_serial.split(' ')[:-1]):
            return x,y
        else:
            raise Exception(txt_serial.split(' ')[0]+"is error!")
    
    def read_batch_txt(batch_txt):
    '''
    读batch的文本
    '''
        batch_x = []
        batch_y = []
        for path in batch_txt:
            if path[-2:]=='\n':
                path = path[:-2]
            x,y = self.read_folder_datasets(path)
            batch_x.append(x)
            batch_y.append(y)
        batch_x=np.transpose(batch_x, (0,4,1,2,3))
        return batch_x,batch_y