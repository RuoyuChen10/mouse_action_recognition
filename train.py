import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataload import Load
from network import C3D


class Train_API:
    def __init__(self):
        self.batch_size = 6
        self.learning_rate = 10e-2
        self.epoches = 100
        self.model = C3D()
        self.load = Load()
        self.training_data_dir='train.txt'
        self.val_data_dir='val.txt'
    def data_init(self):
        '''
        初始化数据
        '''
        train_data_dir,val_data_dir = self.load.read_datasets_txt(self.training_data_dir,self.val_data_dir)
        # train_data, train_label = self.load.read_batch_txt(train_data_dir)
        # val_data, val_label = self.load.read_batch_txt(val_data_dir)
        print("Original data load finished!")
        return train_data_dir,val_data_dir
    def optimize_param(self, model, train_loader, optimizer, criterion):
        '''
        此为更新网络参数
        '''
        model.train()
        for data in train_loader:
            train_data, train_label = self.load.read_batch_txt(data)
            if torch.cuda.is_available():
                train_data = torch.cuda.FloatTensor(train_data)
                train_label = torch.cuda.LongTensor(train_label)
            else:
                train_data = Variable(torch.FloatTensor(train_data))
                train_label = Variable(torch.LongTensor(train_label))
            out = model(train_data)
            loss = criterion(out, train_label)
            print_loss = loss.data.item()
            print("loss=%f"%print_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    def eval_model(self, model, val_loader, criterion):
        '''
        此为监督集的评估
        '''
        model.eval()
        test_loss = 0
        correct = 0
        #with torch.no_grad():
        for data in val_loader:
            val_data, val_label = self.load.read_batch_txt(data)
            if torch.cuda.is_available():
                val_data = torch.cuda.FloatTensor(val_data)
                val_label = torch.cuda.LongTensor(val_label)
            else:
                val_data = Variable(torch.FloatTensor(val_data))
                val_label = Variable(torch.LongTensor(val_label))
            output = model(val_data)
            test_loss += criterion(output, val_label).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(val_label.view_as(pred)).sum().item()
        
        test_loss /= len(val_loader.dataset)
        print('Test set: Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))

    def train(self):
        '''
        训练网络
        '''
        train_data_dir,val_data_dir = self.data_init()
        train_loader = DataLoader(train_data_dir, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data_dir, batch_size=self.batch_size, shuffle=False)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        epoch = 0
        for epoch in range(1, self.epoches + 1):
            self.optimize_param(self.model, train_loader, optimizer, criterion)
            self.eval_model(self.model, val_loader, criterion)
        # 保存
        torch.save(self.model, "./checkpoint/model.pth")  # 保存整个模型

if __name__ == "__main__":
    train = Train_API()
    train.train()