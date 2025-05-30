import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        """时间编码器
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            num_layers: GRU层数
        """
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            
        Returns:
            输出张量 [batch_size, seq_len, hidden_dim]
        """
        output, _ = self.gru(x)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        """空间注意力机制
        
        Args:
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            x: 节点特征 [num_nodes, hidden_dim]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            更新后的节点特征和注意力权重
        """
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)
        attention_weights = F.softmax(self.attention(edge_features), dim=0)
        
        # 聚合邻居信息
        out = torch.zeros_like(x)
        out.index_add_(0, col, x[row] * attention_weights)
        
        return out, attention_weights

class SpatiotemporalFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        """时空融合模块
        
        Args:
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
        """
        super().__init__()
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True
        )
        self.spatial_attention = SpatialAttention(hidden_dim)
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, 
                temporal_features: torch.Tensor,
                spatial_features: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            temporal_features: 时间特征 [batch_size, seq_len, hidden_dim]
            spatial_features: 空间特征 [num_nodes, hidden_dim]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            融合后的特征
        """
        # 时间注意力
        temporal_out, _ = self.temporal_attention(
            temporal_features,
            temporal_features,
            temporal_features
        )
        
        # 空间注意力
        spatial_out, attention_weights = self.spatial_attention(
            spatial_features,
            edge_index
        )
        
        # 特征融合
        fused = torch.cat([temporal_out, spatial_out], dim=-1)
        out = self.fusion_layer(fused)
        
        return out, attention_weights

class TGNNModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_classes: int,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        """T-GNN模型
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_classes: 类别数
            num_layers: 层数
            dropout: Dropout比率
        """
        super().__init__()
        self.temporal_encoder = TemporalEncoder(
            input_dim,
            hidden_dim,
            num_layers
        )
        self.spatial_attention = SpatialAttention(hidden_dim)
        self.fusion = SpatiotemporalFusion(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                time_steps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]
            edge_index: 边索引 [2, num_edges]
            time_steps: 时间步 [batch_size]
            
        Returns:
            预测结果和节点嵌入
        """
        # 时间编码
        temporal_features = self.temporal_encoder(x)
        
        # 空间注意力
        spatial_features, spatial_attention = self.spatial_attention(
            temporal_features[:, -1],  # 使用最后一个时间步
            edge_index
        )
        
        # 时空融合
        fused_features, fusion_attention = self.fusion(
            temporal_features,
            spatial_features,
            edge_index
        )
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits, fused_features

def build_link_features(node_embeddings: torch.Tensor,
                       edge_features: torch.Tensor,
                       link_pairs: torch.Tensor) -> torch.Tensor:
    """构建链路特征
    
    Args:
        node_embeddings: 节点嵌入 [num_nodes, hidden_dim]
        edge_features: 边特征 [num_edges, edge_dim]
        link_pairs: 链路对 [num_links, 2]
        
    Returns:
        链路特征 [num_links, feature_dim]
    """
    # 获取源节点和目标节点的嵌入
    src_emb = node_embeddings[link_pairs[:, 0]]
    dst_emb = node_embeddings[link_pairs[:, 1]]
    
    # 组合特征
    link_features = torch.cat([
        src_emb,
        dst_emb,
        edge_features
    ], dim=1)
    
    return link_features

def temporal_fusion(features_seq: List[torch.Tensor],
                   gamma: float = 0.9) -> torch.Tensor:
    """时间窗口融合
    
    Args:
        features_seq: 特征序列列表
        gamma: 衰减因子
        
    Returns:
        融合后的特征
    """
    if not features_seq:
        return None
    
    # 计算权重
    weights = torch.tensor([
        gamma ** (len(features_seq) - i - 1)
        for i in range(len(features_seq))
    ]).to(features_seq[0].device)
    
    # 加权平均
    weights = weights / weights.sum()
    fused = sum(w * f for w, f in zip(weights, features_seq))
    
    return fused

def tgnn_loss(logits: torch.Tensor,
              labels: torch.Tensor,
              alpha: float = 0.5) -> torch.Tensor:
    """T-GNN损失函数
    
    Args:
        logits: 模型输出 [batch_size, num_classes]
        labels: 真实标签 [batch_size]
        alpha: 平衡因子
        
    Returns:
        损失值
    """
    # 分类损失
    ce_loss = F.cross_entropy(logits, labels)
    
    # 正则化损失
    l2_loss = sum(p.pow(2.0).sum() for p in logits.parameters())
    
    # 总损失
    total_loss = ce_loss + alpha * l2_loss
    
    return total_loss 