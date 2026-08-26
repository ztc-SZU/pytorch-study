import torch
from torch import nn
torch.device('cpu')
torch.device('cuda')
torch.device('cuda:1')
'''
查询GPU数量
'''
print(torch.cuda.device_count())

'''
现在我们定义了两个方便的函数， 这两个函数允许我们在不存在所需所有GPU的情况下运行代码。
'''
def try_gpu(i=0):
    """如果存在,则返回gpu(i),否则返回cpu()"""
    if torch.cuda.device_count()>=i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """返回所有可用的GPU,如果没有GPU,则返回[cpu(),]"""
    devices=[torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
print(try_gpu())
print(try_gpu(10))
print(try_all_gpus())

'''
我们可以查询张量所在的设备。 默认情况下,张量是在CPU上创建的。
'''
x=torch.tensor([1,2,3])
print(x.device)
x=torch.ones(2,3,device=try_gpu())
print(x)

'''
假设我们至少有两个GPU,下面的代码将在第二个GPU上创建一个随机张量。
'''
y=torch.rand(2,3,device=try_gpu(1))
'''
由于Y位于第二个GPU上,所以我们需要将X移到那里, 然后才能执行相加运算。
'''
z=x.cuda(1)

'''
类似地，神经网络模型可以指定设备。 下面的代码将模型参数放在GPU上。
'''
net=nn.Sequential(nn.Linear(3,1))
net=net.to(device=try_gpu())
print(net(x))