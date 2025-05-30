import os
from dataclasses import dataclass

@dataclass
class Config:
    # 数据相关配置
    data_dir: str = 'data'
    output_dir: str = 'outputs'
    batch_size: int = 32
    num_workers: int = 4
    
    # GBDT模型配置
    gbdt_n_estimators: int = 100
    gbdt_learning_rate: float = 0.1
    gbdt_max_depth: int = 5
    gbdt_min_samples_split: int = 2
    gbdt_min_samples_leaf: int = 1
    
    # T-GNN模型配置
    tgnn_hidden_dim: int = 64
    tgnn_output_dim: int = 64
    tgnn_learning_rate: float = 0.001
    tgnn_dropout: float = 0.2
    tgnn_num_layers: int = 2
    
    # 时空融合配置
    fusion_gamma: float = 0.9
    fusion_window_size: int = 5
    
    # 训练配置
    epochs: int = 100
    early_stopping_patience: int = 10
    num_classes: int = 4  # 光纤、卫星、移动、微波
    
    # 设备配置
    device: str = 'cuda'  # 或 'cpu'
    
    def __post_init__(self):
        # 创建必要的目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True) 