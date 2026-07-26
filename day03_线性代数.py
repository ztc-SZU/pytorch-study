import torch
# 标量
x=torch.tensor([3.0])
y=torch.tensor([2.0])
print(x+y)
print(x-y)
print(x*y)
print(x/y)

# 向量
x=torch.arange(4)
print(x)
print(x[3])
print(len(x))
print(x.shape)
A=torch.arange(20).reshape(5,4)
print(A)
# 转置
print(A.T)

# 对称矩阵
B=torch.tensor([[1,2,3],[2,0,4],[3,4,5]])
print(B==B.T)

# clone
A=torch.arange(20,dtype=torch.float32).reshape(5,4)
B=A.clone()
print(A)
print(A+B)

# 矩阵相乘
print(A*B)

# 矩阵与数的运算
print(A+2)
print(A*2)

# mean
print(A)
print(A.sum(axis=0))
print(A.sum(axis=1))
print(A.mean())
print(A.mean(axis=0))
print(A.mean(axis=1))

# 计算总和,平均数时保持轴数不变
sum_A=A.sum(axis=1,keepdims=True)
print(sum_A)

print(A.sum(axis=[0,1]))   # 等价于A.sum()

# 某个轴计算A的累积总和
print(A.cumsum(axis=0))
print(A.cumsum(axis=1))

# 点积
x=torch.tensor([0.0,1,2,3])
y=torch.tensor([1.0,1,1,1])
print(torch.dot(x,y))
print(torch.sum(x*y))

# 矩阵向量积:矩阵×标量  mv
print(A)
print(x)
print(torch.mv(A,x))

# 矩阵*矩阵 mm
print(A)
B=torch.ones(4,3)
print(B)
print(torch.mm(A,B))

# 范数
u=torch.tensor([3.0,4.0])
print(u.norm())