# utils/utils.py 数据加载与预处理
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
def get_data_loaders(batch_size=128, num_workers=2, val_split=0.1):
    """获取 MNIST 数据加载器 （训练集 + 验证集 + 测试集）"""
    
    # 数据预处理：转张量 + 归一化
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST 均值和标准差
    ])

    # 下载训练集和测试集
    full_train_dataset=datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset=datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # ===== 从训练集中切出验证集 =====
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size]
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 创建 DataLoader
    train_loader=DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers
    )
    test_loader=DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return train_loader,val_loader,test_loader

def visualize_samples(dataset, num_samples=16):
    """可视化数据集样本"""
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(num_samples):
        img, label = dataset[i]
        ax = axes[i // 4, i % 4]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f'Label: {label}')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150)
    plt.show()
    print(f"样本图片已保存为 sample_images.png")

def plot_training_history(train_losses, train_accs, val_accs):
    """绘制训练曲线"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(epochs, train_accs, 'r-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'g-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150)
    plt.show()
    print("训练曲线已保存为 training_curve.png")

def plot_confusion_matrix(model, test_loader, device):
    """绘制混淆矩阵"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("混淆矩阵已保存为 confusion_matrix.png")
    
    # 计算各类别准确率
    class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_acc):
        print(f"Digit {i}: {acc:.2%}")
    
    return cm

def visualize_errors(model, test_loader, device, num_errors=10):
    """可视化错误分类的样本"""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # 找出预测错误的样本
            wrong_mask = predicted != labels
            wrong_images = images[wrong_mask]
            wrong_preds = predicted[wrong_mask]
            wrong_labels = labels[wrong_mask]
            
            for img, pred, true in zip(wrong_images, wrong_preds, wrong_labels):
                errors.append((img.cpu(), pred.cpu().item(), true.cpu().item()))
                if len(errors) >= num_errors:
                    break
            if len(errors) >= num_errors:
                break
    
    # 可视化错误样本
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, (img, pred, true) in enumerate(errors):
        ax = axes[i // 5, i % 5]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f'True: {true}, Pred: {pred}', color='red')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('error_samples.png', dpi=150)
    plt.show()
    print("错误样本已保存为 error_samples.png")