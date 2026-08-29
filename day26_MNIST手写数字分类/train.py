# train.py 训练脚本
# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import random
import numpy as np
import numpy as np

from config import Config
from models.model import get_model, count_parameters
from utils.utils import (
    get_data_loaders, 
    plot_training_history, 
    plot_confusion_matrix,
    visualize_errors,
    visualize_samples
)

def set_seed(seed):
    """设置随机种子，保证可复现性"""
    random.seed(seed)  # 1. Python 内置随机数
    np.random.seed(seed)  # 2. NumPy 随机数
    torch.manual_seed(seed)  # 3. PyTorch CPU 随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss=0.0
    correct=0
    total=0

    pbar=tqdm(train_loader,desc='Training')
    for images,labels in pbar:
        images,labels=images.to(device),labels.to(device)

        # 前向传播
        outputs=model(images)
        loss=criterion(outputs,labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss+=loss.item()*images.size(0)
        _,predicted=torch.max(outputs,1)
        total+=images.size(0)
        correct+=(predicted==labels).sum().item()

        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.0 * correct / total:.2f}%'
        })

    epoch_loss=running_loss/total
    epoch_acc=100.0*correct/total
    return epoch_loss,epoch_acc

def evaluate(model, test_loader, criterion, device):
    """评估模型"""
     # 1. 设置评估模式
    model.eval()

    # 2. 初始化统计变量
    running_loss=0.0
    correct=0
    total=0

    # 3. 禁用梯度计算
    with torch.no_grad():
        # 4. 遍历测试数据
        for images,labels in tqdm(test_loader,desc='Evaluating'):
            # 4a. 数据搬到设备
            images,labels=images.to(device),labels.to(device)

            # 4b. 前向传播
            outputs=model(images)
            loss=criterion(outputs,labels)

            # 4c . 统计
            running_loss+=loss.item()*images.size(0)
            _,predicted=torch.max(outputs,1)
            total+=images.size(0)
            correct+=(predicted==labels).sum().item()

            test_loss = running_loss / total
    test_acc = 100.0 * correct / total
    return test_loss, test_acc

def train():
    """主训练函数"""
    # 1. 设置设备和随机种子
    device=torch.device(Config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    set_seed(Config.seed)

     # 2. 加载数据
    print("Loading data...")
    train_loader, val_loader,test_loader=get_data_loaders(
        batch_size=Config.batch_size,
        num_workers=Config.num_workers
    )

    # 3. 创建模型
    print(f"Creating model: {Config.model_type}")
    model=get_model(
        model_type=Config.model_type,
        num_classes=Config.num_classes
    )
    model = model.to(device)
    
    # 打印模型信息
    print(f"Model parameters: {count_parameters(model):,}")

    # 4. 定义损失函数和优化器
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=Config.lr)
    scheduler=ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3)

     # 5. 训练循环
    print("Starting training...")
    train_losses, train_accs, val_accs = [], [], []
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, Config.epochs + 1):
        print(f"\nEpoch {epoch}/{Config.epochs}")
        print("-" * 50)

        # 训练
        train_loss,train_acc=train_one_epoch(model,train_loader,criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 测试
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_accs.append(val_acc)      
         
        # 学习率调度
        scheduler.step(val_loss)

        # 打印结果
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
         
        # 保存最佳模型
        if val_acc>best_val_acc:
            best_val_acc=val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('_')}
            }, os.path.join(Config.save_dir, 'best_model.pth'))
            print(f"Best model saved! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
             
        # 早停
        if patience_counter >= 5:
            print(f"Early stopping at epoch {epoch}")
            break

    # 6. 绘制训练曲线
    plot_training_history(train_losses, train_accs, val_accs)

    
    # 7. 加载最佳模型进行最终评估
    checkpoint = torch.load(os.path.join(Config.save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n Final Test Accuracy: {final_test_acc:.2f}%")
    
    # 8. 混淆矩阵
    plot_confusion_matrix(model, test_loader, device)
    
    # 9. 错误样本可视化
    visualize_errors(model, test_loader, device, num_errors=10)
    
    print("\n Training complete!")
    return model

if __name__ == "__main__":
    # 创建保存目录
    os.makedirs(Config.save_dir, exist_ok=True)
    os.makedirs(Config.log_dir, exist_ok=True)
    
    # 可视化部分训练样本
    train_loader,  val_loader,test_loader = get_data_loaders(batch_size=64)
    visualize_samples(train_loader.dataset, num_samples=16)
    
    # 开始训练
    model = train()