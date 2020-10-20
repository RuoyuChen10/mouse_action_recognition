import numpy as np
import cv2
import os
import torch
import utils

class Load:
    '''
    This class using for .....
    '''
    def __init__(self):
        self.datasets_txt = './dataset.txt'
        self.img_size = (112,112)
        self.num_class = utils.class_num
    def read_datasets_txt(self, training_set_dir="./train.txt",val_set_dir="./val.txt"):
        '''
        读取文本数据
        '''
        training_set_txt = []
        val_set_txt = []
        for line in open(training_set_dir):   
            training_set_txt.append(line)
        for line in open(val_set_dir):   
            val_set_txt.append(line)
        return training_set_txt,val_set_txt
    def read_folder_datasets(self, path):
        '''
        给一行，读一个标签和动作序列
        '''
        y = int(path.split(' ')[-1])
        x = []
        for i in path.split(' ')[:-1]:
            img = cv2.imread(i)
            img = cv2.resize(img,self.img_size)
            img = img/255.
            if img is not None:
                x.append(img)
        if len(x)==len(path.split(' ')[:-1]):
            return x,y
        else:
            raise Exception(path.split(' ')[0]+"is error!")  
    def read_batch_txt(self, batch_txt):
        '''
        读batch的文本
        '''
        batch_x = []
        batch_y = []
        for path in batch_txt:
            if path[-1]=='\n':
                path = path[:-1]
            x,y = self.read_folder_datasets(path)
            batch_x.append(x)
            batch_y.append(y)
        batch_x=np.transpose(batch_x, (0,4,1,2,3))
        # pytorch内部损失函数自动转换为one-hot，不要手动修改
        #batch_y=torch.nn.functional.one_hot(torch.cuda.tensor(batch_y), self.num_class)
        return batch_x, batch_y