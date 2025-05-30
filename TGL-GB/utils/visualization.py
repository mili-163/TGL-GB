import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import torch

def plot_training_curves(train_losses, val_losses, output_dir):
    """绘制训练和验证损失曲线
    
    Args:
        train_losses (list): 训练损失列表
        val_losses (list): 验证损失列表
        output_dir (str): 输出目录
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失曲线')
    plt.legend()
    plt.grid(True)
    
    # 保存图片
    output_path = Path(output_dir) / 'plots' / 'training_curves.png'
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(confusion_matrix, class_names, output_dir):
    """绘制混淆矩阵
    
    Args:
        confusion_matrix (np.ndarray): 混淆矩阵
        class_names (list): 类别名称列表
        output_dir (str): 输出目录
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    
    # 保存图片
    output_path = Path(output_dir) / 'plots' / 'confusion_matrix.png'
    plt.savefig(output_path)
    plt.close()

def plot_feature_importance(feature_importance, feature_names, output_dir):
    """绘制特征重要性图
    
    Args:
        feature_importance (np.ndarray): 特征重要性数组
        feature_names (list): 特征名称列表
        output_dir (str): 输出目录
    """
    plt.figure(figsize=(12, 6))
    indices = np.argsort(feature_importance)
    plt.barh(range(len(indices)), feature_importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('特征重要性')
    plt.title('特征重要性排序')
    
    # 保存图片
    output_path = Path(output_dir) / 'plots' / 'feature_importance.png'
    plt.savefig(output_path)
    plt.close()

def plot_attention_weights(attention_weights, output_dir):
    """绘制注意力权重图
    
    Args:
        attention_weights (torch.Tensor): 注意力权重张量
        output_dir (str): 输出目录
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights.detach().cpu().numpy(),
        cmap='YlOrRd',
        annot=True,
        fmt='.2f'
    )
    plt.xlabel('目标节点')
    plt.ylabel('源节点')
    plt.title('注意力权重分布')
    
    # 保存图片
    output_path = Path(output_dir) / 'plots' / 'attention_weights.png'
    plt.savefig(output_path)
    plt.close()

def plot_temporal_patterns(temporal_data, output_dir):
    """绘制时间模式图
    
    Args:
        temporal_data (np.ndarray): 时间序列数据
        output_dir (str): 输出目录
    """
    plt.figure(figsize=(12, 6))
    for i in range(temporal_data.shape[1]):
        plt.plot(temporal_data[:, i], label=f'特征 {i+1}')
    plt.xlabel('时间步')
    plt.ylabel('特征值')
    plt.title('时间模式分析')
    plt.legend()
    plt.grid(True)
    
    # 保存图片
    output_path = Path(output_dir) / 'plots' / 'temporal_patterns.png'
    plt.savefig(output_path)
    plt.close() 