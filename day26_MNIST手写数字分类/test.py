# test.py 测试脚本
# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch
import os
from config import Config
from models.model import get_model
from train import evaluate
from utils.utils import get_data_loaders, plot_confusion_matrix

def test():
    device=torch.device(Config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model = get_model(Config.model_type, Config.num_classes)
    checkpoint = torch.load(os.path.join(Config.save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 加载数据
    _,_,test_loader = get_data_loaders(
        batch_size=Config.batch_size,
        num_workers=Config.num_workers
    )

    # 计算测试准确率  
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n Final Test Accuracy: {test_acc:.2f}%")
    # 混淆矩阵
    plot_confusion_matrix(model, test_loader, device)

if __name__ == "__main__":
    test()