import numpy as np
import pandas as pd
from typing import Dict

class FeatureInitializer:
    def __init__(self):
        """初始化特征初始化器"""
        self.static_feature_dim = None
        self.dynamic_feature_dim = None
        self.edge_feature_dim = None
        
    def _get_static_features(self, node_data: pd.DataFrame) -> np.ndarray:
        """
        获取节点的静态特征
        
        参数:
        node_data: 包含节点静态特征的DataFrame
            
        返回:
        static_features: 节点的静态特征矩阵
        """
        # 提取静态特征（IP地址、ASN、ISP等）
        static_features = node_data[['ip_address', 'asn', 'isp']].values
        return static_features
        
    def _get_dynamic_features(self, edge_data: pd.DataFrame) -> np.ndarray:
        """
        获取边的动态特征
        
        参数:
        edge_data: 包含边动态特征的DataFrame
            
        返回:
        dynamic_features: 边的动态特征矩阵
        """
        # 提取动态特征（RTT统计特征和多项式拟合系数）
        dynamic_features = edge_data[['min_rtt', 'var_rtt']].values
        coefficients = np.array([coef for coef in edge_data['coefficients']])
        return np.hstack((dynamic_features, coefficients))
        
    def _get_edge_features(self, edge_data: pd.DataFrame) -> np.ndarray:
        """
        获取边的特征
        
        参数:
        edge_data: 包含边特征的DataFrame
            
        返回:
        edge_features: 边的特征矩阵
        """
        # 提取时间序列延迟特征
        delay_features = edge_data[['timestamp', 'rtt']].values
        
        # 提取RTT统计特征
        rtt_features = edge_data[['min_rtt', 'var_rtt']].values
        
        # 合并所有边特征
        edge_features = np.hstack((delay_features, rtt_features))
        return edge_features
        
    def initialize_features(self, 
                          head_nodes: pd.DataFrame,
                          tail_nodes: pd.DataFrame,
                          edges: pd.DataFrame) -> np.ndarray:
        """
        初始化特征矩阵
        
        参数:
        head_nodes: 头节点的特征数据
        tail_nodes: 尾节点的特征数据
        edges: 边的特征数据
            
        返回:
        feature_matrix: 完整的特征矩阵
        """
        # 获取头节点的静态特征
        head_static = self._get_static_features(head_nodes)
        self.static_feature_dim = head_static.shape[1]
        
        # 获取尾节点的静态特征
        tail_static = self._get_static_features(tail_nodes)
        
        # 获取边的动态特征
        edge_dynamic = self._get_dynamic_features(edges)
        self.dynamic_feature_dim = edge_dynamic.shape[1]
        
        # 获取边的特征
        edge_features = self._get_edge_features(edges)
        self.edge_feature_dim = edge_features.shape[1]
        
        # 构建完整的特征矩阵
        feature_matrix = np.hstack((
            head_static,  # 头节点静态特征
            edge_dynamic,  # 边动态特征
            edge_features,  # 边特征
            tail_static  # 尾节点静态特征
        ))
        
        return feature_matrix
        
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        获取各个特征的维度信息
        
        返回:
        dimensions: 包含各个特征维度的字典
        """
        return {
            'static_feature_dim': self.static_feature_dim,
            'dynamic_feature_dim': self.dynamic_feature_dim,
            'edge_feature_dim': self.edge_feature_dim,
            'total_feature_dim': (self.static_feature_dim * 2 + 
                                self.dynamic_feature_dim + 
                                self.edge_feature_dim)
        } 