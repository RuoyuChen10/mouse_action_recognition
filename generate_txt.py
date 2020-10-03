import numpy
import os
import random

datasets_root = './datasets'    #数据集路径
serial_num = 16                 #一个动作序列数量
training_set = "train.txt"      #训练集存放
val_set = "val.txt"             #监督集存放
test_set = "test.txt"           #测试集存放

def read_sort_file(file_dir):
    '''
    文本排序
    '''
    try:
        file_list = os.listdir(file_dir)
        file_list.sort(key=lambda x: int(x.split('.')[0]))
        return file_list
    except:
        return None

def del_file(file_name):
    '''
    删除路径下文件
    '''
    if os.path.exists(file_name):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
        os.remove(file_name)  

def generate(rootfile,txt_name="datasets.txt",num=10):
    '''
    生成datasets.txt文件
    '''
    del_file(txt_name)
    files_dir = read_sort_file(rootfile)
    for action_class in files_dir:
        class_dir = read_sort_file(rootfile+'/'+action_class)
        for i in class_dir:
            image_name = read_sort_file(rootfile+'/'+action_class+'/'+i)
            try:
                if(len(image_name)>=num):
                    for j in range(0,num): 
                        with open(txt_name,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
                            file.write(rootfile+'/'+action_class +'/'+i+'/'+image_name[j]+' ')
                    with open(txt_name,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
                        file.write(action_class+"\n") 
            except:
                pass

def distribute(txt_name="datasets.txt"):
    '''
    随机打乱数据集顺序，输入为生成的整齐的txt文本
    '''
    list = []
    for line in open("datasets.txt"):   
        list.append(line)
    random.shuffle(list)
    del_file(txt_name)
    for i in list:
        with open(txt_name,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
            file.write(i)

def split_datasets(txt_name="datasets.txt",train_rate=0.8,val_rate=0.1,test_rate=0.1):
    '''
    通过生成的datatsets文本，分为train,val,和test集
    '''
    list = []
    # 清除三个数据集
    del_file(training_set)
    del_file(val_set)
    del_file(test_set)
    # 读取生成的文件
    for line in open(txt_name):   
        list.append(line)
    num = len(list)
    # 计算随机分配的数量
    train_num = int(num*train_rate)
    val_num = int(num*val_rate)
    test_num = num-train_num-val_num
    # 依次写入文本
    for i in range(0,train_num):
        with open(training_set,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
            file.write(list[i])
    for i in range(train_num,train_num + val_num):
        with open(val_set,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
            file.write(list[i])
    for i in range(train_num + val_num,num):
        with open(test_set,"a") as file:   #只需要将之前的”w"改为“a"即可，代表追加内容
            file.write(list[i])

if __name__ == '__main__':
    generate(rootfile=datasets_root,txt_name="datasets.txt",num=serial_num)
    distribute(txt_name="datasets.txt")
    split_datasets()

