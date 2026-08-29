# config.py   配置文件
class Config:
    # 数据参数
    batch_size = 128
    num_workers = 2
    
    # 训练参数
    epochs = 20
    lr = 0.001
    device = 'cuda'  
    
    # 模型参数
    model_type = 'CNN'  # 'MLP', 'LeNet', 'CNN'
    num_classes = 10
    
    # 保存路径
    save_dir = './checkpoints' # 模型权重保存路径
    log_dir = './logs' # 训练日志保存路径
    
    # 随机种子
    seed = 42