# models/model.py  模型定义
import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self,input_size=784,hidden_size=256,num_classes=10):
        super().__init__()
        self.flatten=nn.Flatten()
        self.fc1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.2) # 训练时随机丢弃 20% 的神经元，防止过拟合。
        self.fc2=nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        x=self.flatten(x)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x

class LeNet(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()
        self.flatten=nn.Flatten()
    def forward(self,x):
        x=self.pool1(self.relu(self.conv1(x)))
        x=self.pool2(self.relu(self.conv2(x)))
        x=self.flatten(x)
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(0.25)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.flatten=nn.Flatten()
        self.relu=nn.ReLU()

    def forward(self, x):
        # 输入: (batch, 1, 28, 28)
        x=self.pool(self.relu(self.bn1(self.conv1(x))))  # (batch, 32, 14, 14)
        x=self.pool(self.relu(self.bn2(self.conv2(x))))  # (batch, 64, 7, 7)
        x=self.pool(self.relu(self.bn3(self.conv3(x))))  # (batch, 128, 3, 3)

        x=self.flatten(x)
        x=self.dropout(self.relu(self.fc1(x)))
        x=self.fc2(x)
        return x

def get_model(model_type='CNN', num_classes=10):
    """工厂函数：根据类型返回模型"""
    if model_type == 'MLP':
        return MLP(num_classes=num_classes)
    elif model_type == 'LeNet':
        return LeNet(num_classes=num_classes)
    elif model_type == 'CNN':
        return CNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# 模型参数量统计
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)