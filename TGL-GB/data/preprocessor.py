import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, Dict, List
import torch

class DataPreprocessor:
    def __init__(self):
        """初始化数据预处理器"""
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse=False)
        self.feature_names = None
        
    def preprocess_static_features(self, 
                                 ip_addresses: List[str],
                                 asn_numbers: List[int],
                                 isp_info: List[str]) -> np.ndarray:
        """预处理静态特征
        
        Args:
            ip_addresses: IP地址列表
            asn_numbers: ASN编号列表
            isp_info: ISP信息列表
            
        Returns:
            处理后的特征矩阵
        """
        # 处理IP地址
        ip_features = self._process_ip_addresses(ip_addresses)
        
        # 处理ASN编号
        asn_features = self._process_asn_numbers(asn_numbers)
        
        # 处理ISP信息
        isp_features = self._process_isp_info(isp_info)
        
        # 合并特征
        static_features = np.hstack([ip_features, asn_features, isp_features])
        
        return static_features
    
    def preprocess_dynamic_features(self,
                                  rtt_data: np.ndarray,
                                  hop_counts: np.ndarray) -> np.ndarray:
        """预处理动态特征
        
        Args:
            rtt_data: RTT数据 [num_samples, num_timesteps, num_hops]
            hop_counts: 跳数数据 [num_samples, num_timesteps]
            
        Returns:
            处理后的特征矩阵
        """
        num_samples = rtt_data.shape[0]
        num_timesteps = rtt_data.shape[1]
        
        # 初始化特征列表
        dynamic_features = []
        
        for i in range(num_samples):
            sample_features = []
            
            for t in range(num_timesteps):
                # 提取当前时间步的RTT和跳数
                rtt = rtt_data[i, t]
                hops = hop_counts[i, t]
                
                # 计算RTT统计特征
                rtt_stats = self._calculate_rtt_statistics(rtt, hops)
                
                # 计算RTT趋势特征
                rtt_trends = self._calculate_rtt_trends(rtt, hops)
                
                # 合并时间步特征
                timestep_features = np.concatenate([
                    rtt_stats,
                    rtt_trends
                ])
                
                sample_features.append(timestep_features)
            
            # 堆叠时间步特征
            sample_features = np.stack(sample_features)
            dynamic_features.append(sample_features)
        
        # 堆叠样本特征
        dynamic_features = np.stack(dynamic_features)
        
        return dynamic_features
    
    def _process_ip_addresses(self, ip_addresses: List[str]) -> np.ndarray:
        """处理IP地址特征
        
        Args:
            ip_addresses: IP地址列表
            
        Returns:
            IP地址特征矩阵
        """
        features = []
        for ip in ip_addresses:
            # 分割IP地址
            octets = list(map(int, ip.split('.')))
            
            # 提取特征
            ip_features = [
                octets[0],  # 第一个八位字节
                octets[1],  # 第二个八位字节
                octets[2],  # 第三个八位字节
                octets[3],  # 第四个八位字节
                sum(octets),  # 总和
                np.mean(octets),  # 平均值
                np.std(octets)  # 标准差
            ]
            features.append(ip_features)
        
        return np.array(features)
    
    def _process_asn_numbers(self, asn_numbers: List[int]) -> np.ndarray:
        """处理ASN编号特征
        
        Args:
            asn_numbers: ASN编号列表
            
        Returns:
            ASN编号特征矩阵
        """
        # 将ASN编号转换为分类特征
        asn_features = self.encoder.fit_transform(
            np.array(asn_numbers).reshape(-1, 1)
        )
        
        return asn_features
    
    def _process_isp_info(self, isp_info: List[str]) -> np.ndarray:
        """处理ISP信息特征
        
        Args:
            isp_info: ISP信息列表
            
        Returns:
            ISP信息特征矩阵
        """
        # 将ISP信息转换为分类特征
        isp_features = self.encoder.fit_transform(
            np.array(isp_info).reshape(-1, 1)
        )
        
        return isp_features
    
    def _calculate_rtt_statistics(self,
                                rtt: np.ndarray,
                                hops: np.ndarray) -> np.ndarray:
        """计算RTT统计特征
        
        Args:
            rtt: RTT数据 [num_hops]
            hops: 跳数数据 [num_hops]
            
        Returns:
            RTT统计特征
        """
        # 基本统计量
        mean_rtt = np.mean(rtt)
        std_rtt = np.std(rtt)
        min_rtt = np.min(rtt)
        max_rtt = np.max(rtt)
        
        # 每跳RTT
        rtt_per_hop = rtt / (hops + 1e-6)  # 避免除零
        
        # RTT变化率
        rtt_changes = np.diff(rtt)
        mean_change = np.mean(rtt_changes)
        std_change = np.std(rtt_changes)
        
        # 组合特征
        stats = np.array([
            mean_rtt,
            std_rtt,
            min_rtt,
            max_rtt,
            np.mean(rtt_per_hop),
            np.std(rtt_per_hop),
            mean_change,
            std_change
        ])
        
        return stats
    
    def _calculate_rtt_trends(self,
                            rtt: np.ndarray,
                            hops: np.ndarray) -> np.ndarray:
        """计算RTT趋势特征
        
        Args:
            rtt: RTT数据 [num_hops]
            hops: 跳数数据 [num_hops]
            
        Returns:
            RTT趋势特征
        """
        # 计算RTT随跳数的变化趋势
        if len(rtt) > 1:
            # 线性回归系数
            slope, _ = np.polyfit(hops, rtt, 1)
            
            # 二次多项式系数
            poly_coeffs = np.polyfit(hops, rtt, 2)
            
            # 计算RTT的波动性
            rtt_diff = np.diff(rtt)
            volatility = np.std(rtt_diff)
            
            # 计算RTT的稳定性
            stability = 1.0 / (1.0 + volatility)
        else:
            slope = 0
            poly_coeffs = np.zeros(3)
            volatility = 0
            stability = 1
        
        # 组合特征
        trends = np.array([
            slope,
            poly_coeffs[0],
            poly_coeffs[1],
            volatility,
            stability
        ])
        
        return trends
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """标准化特征
        
        Args:
            features: 特征矩阵
            
        Returns:
            标准化后的特征矩阵
        """
        return self.scaler.fit_transform(features)
    
    def create_edge_index(self, num_nodes: int) -> torch.Tensor:
        """创建边索引
        
        Args:
            num_nodes: 节点数量
            
        Returns:
            边索引张量 [2, num_edges]
        """
        # 创建完全图
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edges.append([i, j])
                edges.append([j, i])  # 无向图需要双向边
        
        return torch.tensor(edges, dtype=torch.long).t()
    
    def create_link_pairs(self, num_nodes: int) -> torch.Tensor:
        """创建链路对
        
        Args:
            num_nodes: 节点数量
            
        Returns:
            链路对张量 [num_links, 2]
        """
        # 创建所有可能的链路对
        pairs = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                pairs.append([i, j])
        
        return torch.tensor(pairs, dtype=torch.long) 