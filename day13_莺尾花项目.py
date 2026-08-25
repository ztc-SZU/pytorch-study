import torch
from torch import nn
from torch.utils import data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
# 1.加载数据
iris=load_iris()
X=iris.data
y=iris.target

# 2.划分训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# 3.特征标准化
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 4.将数据转换为PyTorch张量
X_train_t=torch.tensor(X_train,dtype=torch.float32)
X_test_t=torch.tensor(X_test,dtype=torch.float32)
y_train_t=torch.tensor(y_train,dtype=torch.long)
y_test_t=torch.tensor(y_test,dtype=torch.long)

# 5.创建数据加载器
batch_size=16
train_dataset=data.TensorDataset(X_train_t,y_train_t)
test_dataset=data.TensorDataset(X_test_t,y_test_t)
train_iter=data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_iter=data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

# 6.定义模型
net=nn.Sequential(
    nn.Linear(4,64),
    nn.ReLU(),
    nn.Linear(64,3)
)
def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # 适合 ReLU
        nn.init.zeros_(m.bias)
net.apply(init_weights)

# 7.定义损失函数和优化器
loss=nn.CrossEntropyLoss()
lr=0.01
trainer=torch.optim.Adam(net.parameters(),lr=lr)

# 8.训练模型
num_epochs=200
train_losses=[]
test_accs=[]
for epoch in range(num_epochs):
    net.train() # 转换为训练模式
    total_loss=0
    for X_batch,y_batch in train_iter:
        y_hat=net(X_batch)
        l=loss(y_hat,y_batch)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        total_loss+=l.item()*X_batch.size(0)

    # 每个epoch结束后计算测试准确率
    net.eval() # 转换为评估模式
    correct=0
    total=0
    with torch.no_grad():
        for X_batch,y_batch in test_iter:
            y_hat=net(X_batch)
            pred=y_hat.argmax(dim=1)
            correct+=(pred==y_batch).sum().item()
            total+=y_batch.size(0)
    test_acc = correct / total
    train_losses.append(total_loss / len(train_dataset))
    test_accs.append(test_acc)

    # 每20个epoch打印一次
    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_losses[-1]:.4f}, Test Acc: {test_acc:.4f}')

plt.figure(figsize=(10, 4))
# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Loss Curve')

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(test_accs)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Accuracy Curve')
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# 8.测试集评估
net.eval()
with torch.no_grad():
    y_hat = net(X_test_t)
    pred = y_hat.argmax(dim=1)
    final_acc = (pred == y_test_t).float().mean().item()
    print(f'最终测试集准确率: {final_acc:.4f}')