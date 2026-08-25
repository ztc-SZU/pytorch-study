import torch
x=torch.arange(4.0,requires_grad=True)
y=2*torch.dot(x,x)
print(y)

# 通过调用反向传播函数,自动计算y关于x每个分量的梯度
y.backward()
print(x.grad)
print(x.grad==4*x)

# 默认情况下PyTorch
x.grad.zero_()
y=x.sum()
y.backward()
print(x.grad)

x.grad.zero_()
y=x*x
y.sum().backward()
print(x.grad)

# 将某些计算移到记录的计算图之外
x.grad.zero_()
y=x*x
u=y.detach()  # 变成一个与x无关的变量
z=u*x
z.sum().backward()
print(x.grad)