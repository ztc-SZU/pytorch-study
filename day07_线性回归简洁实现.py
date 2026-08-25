import numpy as np
import torch
from torch.utils import data
from torch import nn
def synthetic_data(w,b,num_examples):
    """生成y=Xw+b+噪声。"""
    X=torch.normal(0,1,(num_examples,len(w)))
    y=torch.matmul(X,w)+b
    y+=torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))
true_w=torch.tensor([2,-3.4])
true_b=4.2
# 生成数据集
features,labels=synthetic_data(true_w,true_b,1000)

# 读取数据
def load_array(data_arrays,batch_size,is_train=True):
    """构造一个PyTroch数据迭代器"""
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
batch_size=10
data_iter=load_array((features,labels),batch_size)
next(iter(data_iter))

# 使用框架预定于好的层，nn是神经网络的缩写
net=nn.Sequential(nn.Linear(2,1))

# 初始化模型参数
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

# 计算均方误差
loss=nn.MSELoss()

# 实例化SGD
trainer=torch.optim.SGD(net.parameters(),lr=0.03)

num_epochs=3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l=loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l=loss(net(features),labels)
    print(f"epoch:{epoch+1},loss:{l}")