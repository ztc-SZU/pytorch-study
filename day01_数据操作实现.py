import torch
# 1.生成一个数组
x=torch.arange(12)
print(x)

# 2.形状和总数
print(x.shape)
print(x.numel())

# 3.改变形状
X=x.reshape(3,4)
print(X)

# 4.全0,全1
q=torch.zeros(1,2,3)
print(q)
q=torch.ones(1,2,3)
print(q)

# 5.列表赋值
x=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(x)

# 6.常见算术运算符
x=torch.tensor([1.0,2,3])
y=torch.tensor([4,5,6])
print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x**y)
# 指数
q=torch.exp(x)
print(q)

# cat拼接
x=torch.arange(12,dtype=torch.float32).reshape(3,4)
y=torch.tensor([[1.0,2,3,4],[5,6,7,8],[10,11,12,13]])
X=torch.cat((x,y),dim=0) # 按行拼接
Y=torch.cat((x,y),dim=1) # 按列拼接
print(X)
print(Y)

# ==
print(x==y)

# sum
print(x.sum())

# 广播机制
x=torch.arange(3).reshape((3,1))
y=torch.arange(2).reshape((1,2))
print(x+y)

# -1访问最后一行元素
x=torch.arange(12,dtype=torch.float32).reshape(3,4)
print(x[-1])
print(x[1:3])

# 通过指定索引，将元素写入矩阵
x[1,2]=9