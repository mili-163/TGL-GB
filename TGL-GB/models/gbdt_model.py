import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from lightgbm import LGBMClassifier

class LeafEmbeddingMLP(nn.Module):
    """用于将叶子节点索引嵌入到密集空间的MLP"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class GBDTModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5):
        """初始化GBDT模型
        
        Args:
            n_estimators (int): 树的数量
            learning_rate (float): 学习率
            max_depth (int): 树的最大深度
        """
        self.gbdt = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=2**max_depth,
            objective='multiclass',
            random_state=42
        )
        self.encoder = OneHotEncoder(sparse=False)
        self.leaf_embedding = None
        
    def fit(self, X, y):
        """训练GBDT模型
        
        Args:
            X (np.ndarray): 特征矩阵
            y (np.ndarray): 标签
        """
        self.gbdt.fit(X, y)
        
        # 获取叶子节点索引
        leaf_indices = self.gbdt.predict(X, pred_leaf=True)
        
        # 对叶子节点进行OneHot编码
        self.encoder.fit(leaf_indices)
        leaf_embeddings = self.encoder.transform(leaf_indices)
        
        # 创建叶子节点嵌入层
        self.leaf_embedding = nn.Linear(leaf_embeddings.shape[1], 64)
        
        return self
        
    def predict(self, X):
        """预测并返回叶子节点嵌入
        
        Args:
            X (np.ndarray): 特征矩阵
            
        Returns:
            tuple: (预测标签, 叶子节点嵌入)
        """
        # 获取预测标签
        y_pred = self.gbdt.predict(X)
        
        # 获取叶子节点索引
        leaf_indices = self.gbdt.predict(X, pred_leaf=True)
        
        # 对叶子节点进行OneHot编码
        leaf_embeddings = self.encoder.transform(leaf_indices)
        
        # 通过嵌入层
        if self.leaf_embedding is not None:
            leaf_embeddings = self.leaf_embedding(torch.FloatTensor(leaf_embeddings))
        
        return y_pred, leaf_embeddings
    
    def get_feature_importance(self):
        """获取特征重要性
        
        Returns:
            np.ndarray: 特征重要性数组
        """
        return self.gbdt.feature_importances_
    
    def save_model(self, path):
        """保存模型
        
        Args:
            path (str): 保存路径
        """
        import joblib
        joblib.dump({
            'gbdt': self.gbdt,
            'encoder': self.encoder,
            'leaf_embedding': self.leaf_embedding
        }, path)
    
    def load_model(self, path):
        """加载模型
        
        Args:
            path (str): 模型路径
        """
        import joblib
        model_dict = joblib.load(path)
        self.gbdt = model_dict['gbdt']
        self.encoder = model_dict['encoder']
        self.leaf_embedding = model_dict['leaf_embedding']

    def _build_tree(self, X: np.ndarray, residuals: np.ndarray) -> DecisionTreeRegressor:
        """
        构建单个决策树
        
        参数:
        X: 特征矩阵
        residuals: 残差
            
        返回:
        tree: 训练好的决策树
        """
        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split
        )
        tree.fit(X, residuals)
        return tree
        
    def _get_leaf_indices(self, X: np.ndarray) -> np.ndarray:
        """
        获取样本在所有树中的叶子节点索引
        
        参数:
        X: 特征矩阵
            
        返回:
        leaf_indices: 叶子节点索引矩阵
        """
        leaf_indices = []
        for tree in self.trees:
            indices = tree.apply(X)
            leaf_indices.append(indices)
        return np.column_stack(leaf_indices)
        
    def _encode_leaf_indices(self, leaf_indices: np.ndarray) -> torch.Tensor:
        """
        对叶子节点索引进行编码和嵌入
        
        参数:
        leaf_indices: 叶子节点索引矩阵
            
        返回:
        embedded_features: 嵌入后的特征
        """
        # 对每棵树的叶子节点进行one-hot编码
        encoded_features = []
        for t, indices in enumerate(leaf_indices.T):
            if len(self.leaf_encoders) <= t:
                encoder = OneHotEncoder(sparse_output=False)
                encoder.fit(indices.reshape(-1, 1))
                self.leaf_encoders.append(encoder)
            encoded = self.leaf_encoders[t].transform(indices.reshape(-1, 1))
            encoded_features.append(encoded)
            
        # 将所有编码后的特征连接起来
        concatenated = np.hstack(encoded_features)
        
        # 使用MLP进行降维
        embedded = self.leaf_embedding(torch.FloatTensor(concatenated))
        return embedded
        
    def update_features(self, 
                       X: np.ndarray, 
                       tgnn_gradients: np.ndarray,
                       learning_rate: float) -> np.ndarray:
        """
        根据T-GNN的梯度更新特征
        """
        self.fit(X, np.zeros_like(tgnn_gradients), tgnn_gradients)
        updated_features = X - learning_rate * tgnn_gradients
        return updated_features 