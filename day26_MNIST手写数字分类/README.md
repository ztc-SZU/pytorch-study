# MNIST 手写数字分类

基于 PyTorch 实现的 MNIST 手写数字识别项目，包含 MLP、LeNet 和 CNN 三种模型。

---

## 📁 项目结构
day26_MNIST手写数字分类/
├── config.py # 配置文件
├── train.py # 训练脚本
├── test.py # 测试脚本
├── models/
│ └── model.py # 模型定义 (MLP, LeNet, CNN)
├── utils/
│ └── utils.py # 工具函数 (数据加载、可视化)
├── data/ # MNIST 数据集 (自动下载)
├── checkpoints/ # 保存的模型权重
│ └── best_model.pth
├── logs/ # 训练日志
├── requirements.txt # 依赖包列表
└── README.md # 项目说明

text
---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt

2. 训练模型
python train.py

3. 测试模型
python test.py

📊 模型性能
模型	参数量	测试准确率
MLP	~203K	~98.41%
LeNet	~62K	~99.17%
CNN (推荐)	~391K	99.36%
各类别准确率
数字	准确率	数字	准确率
0	99.59%	5	99.44%
1	99.82%	6	98.75%
2	99.42%	7	99.71%
3	99.60%	8	99.18%
4	99.29%	9	98.71%
🖼️ 可视化结果
训练完成后会生成以下图片：

图片	说明
sample_images.png	训练样本预览
training_curve.png	损失和准确率曲线
confusion_matrix.png	混淆矩阵
error_samples.png	错误分类样本
🛠️ 配置说明
在 config.py 中调整训练参数：

python
class Config:
    batch_size = 128      # 批次大小
    epochs = 20           # 训练轮数
    lr = 0.001            # 学习率
    model_type = 'CNN'    # 模型类型: 'MLP', 'LeNet', 'CNN'
    device = 'cuda'       # 训练设备: 'cuda' 或 'cpu'
📦 依赖
Python 3.8+

PyTorch 1.10+

torchvision

matplotlib

seaborn

scikit-learn

tqdm

numpy

📝 训练日志示例
text
Using device: cuda
Loading data...
训练集大小: 54000
验证集大小: 6000
测试集大小: 10000
Creating model: CNN
Model parameters: 390,858
Starting training...

Epoch 8/20
--------------------------------------------------
Training: 100% | 422/422 [00:09<00:00, 44.83it/s, Loss=0.0098, Acc=99.83%]
Evaluating: 100% | 47/47 [00:07<00:00, 6.36it/s]
Train Loss: 0.0061 | Train Acc: 99.83%
Val Loss: 0.0272 | Val Acc: 99.32%
Best model saved! (Val Acc: 99.32%)

==================================================
Final Test Accuracy: 99.35%
==================================================
📚 参考
PyTorch 官方文档

MNIST 数据集

动手学深度学习

