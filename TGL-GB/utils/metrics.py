import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def calculate_metrics(y_true, y_pred, average='weighted'):
    """计算分类性能指标
    
    Args:
        y_true (np.ndarray): 真实标签
        y_pred (np.ndarray): 预测标签
        average (str, optional): 多分类评估的平均方式
    
    Returns:
        dict: 包含各项指标的字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
        'f1': f1_score(y_true, y_pred, average=average)
    }
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算每个类别的指标
    class_metrics = {}
    for i in range(len(np.unique(y_true))):
        class_metrics[f'class_{i}'] = {
            'precision': precision_score(y_true, y_pred, labels=[i], average='micro'),
            'recall': recall_score(y_true, y_pred, labels=[i], average='micro'),
            'f1': f1_score(y_true, y_pred, labels=[i], average='micro')
        }
    
    metrics['class_metrics'] = class_metrics
    metrics['confusion_matrix'] = cm
    
    return metrics

def print_metrics(metrics):
    """打印评估指标
    
    Args:
        metrics (dict): 评估指标字典
    """
    print("\n=== 模型评估指标 ===")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")
    
    print("\n=== 各类别指标 ===")
    for class_name, class_metrics in metrics['class_metrics'].items():
        print(f"\n{class_name}:")
        print(f"  精确率: {class_metrics['precision']:.4f}")
        print(f"  召回率: {class_metrics['recall']:.4f}")
        print(f"  F1分数: {class_metrics['f1']:.4f}")
    
    print("\n=== 混淆矩阵 ===")
    print(metrics['confusion_matrix']) 