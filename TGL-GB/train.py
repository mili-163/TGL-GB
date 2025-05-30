import os
import torch
import numpy as np
import logging
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from data.data_loader import DataLoader as CustomDataLoader
from models.gbdt_model import GBDTModel
from models.tgnn import (
    TGNNModel, 
    build_link_features, 
    TemporalFusionNetwork, 
    temporal_fusion, 
    tgnn_loss,
    SpatiotemporalFusion
)
from utils.logger import setup_logger
from utils.config import Config
from utils.metrics import calculate_metrics
from utils.visualization import plot_training_curves

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志记录"""
        log_dir = os.path.join(self.config.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = setup_logger(
            name='trainer',
            log_file=os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        )
        
    def prepare_data(self):
        """准备训练数据"""
        self.logger.info("开始加载数据...")
        data_loader = CustomDataLoader()
        X_train, X_test, y_train, y_test = data_loader.load_all_data()
        
        # 数据预处理
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        self.logger.info(f"数据加载完成。训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        
    def train_gbdt(self):
        """训练GBDT模型"""
        self.logger.info("开始训练GBDT模型...")
        self.gbdt = GBDTModel(
            n_estimators=self.config.gbdt_n_estimators,
            learning_rate=self.config.gbdt_learning_rate,
            max_depth=self.config.gbdt_max_depth
        )
        
        # 训练GBDT
        self.gbdt.fit(self.X_train, self.y_train)
        
        # 获取叶子节点嵌入
        _, gbdt_embedded = self.gbdt.predict(self.X_train)
        self.X_enhanced = np.hstack([self.X_train, gbdt_embedded.detach().numpy()])
        
        self.logger.info("GBDT模型训练完成")
        
    def train_tgnn(self):
        """训练T-GNN模型"""
        self.logger.info("开始训练T-GNN模型...")
        
        # 初始化T-GNN模型
        self.tgnn = TGNNModel(
            input_dim=self.X_enhanced.shape[1],
            hidden_dim=self.config.tgnn_hidden_dim,
            output_dim=self.config.tgnn_output_dim,
            num_classes=self.config.num_classes
        ).to(self.device)
        
        # 初始化优化器
        optimizer = torch.optim.Adam(
            self.tgnn.parameters(),
            lr=self.config.tgnn_learning_rate
        )
        
        # 训练循环
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.epochs):
            self.tgnn.train()
            epoch_loss = 0
            
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                logits, node_emb = self.tgnn(batch_X, self.edge_index, self.time_steps)
                
                # 构建链路特征
                link_features = build_link_features(
                    node_emb,
                    self.edge_features,
                    self.link_pairs
                )
                
                # 时间窗口融合
                link_features_seq = [link_features]
                fused_features = temporal_fusion(
                    link_features_seq,
                    gamma=self.config.fusion_gamma
                )
                
                # TFN分类
                y_pred = self.tfn(fused_features)
                
                # 计算损失
                loss = tgnn_loss(y_pred, batch_y)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # 验证
            val_loss = self.validate()
            train_losses.append(epoch_loss / len(self.train_loader))
            val_losses.append(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch)
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - "
                f"Train Loss: {train_losses[-1]:.4f} - "
                f"Val Loss: {val_loss:.4f}"
            )
        
        # 绘制训练曲线
        plot_training_curves(train_losses, val_losses, self.config.output_dir)
        
    def validate(self):
        """验证模型性能"""
        self.tgnn.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                logits, node_emb = self.tgnn(batch_X, self.edge_index, self.time_steps)
                link_features = build_link_features(
                    node_emb,
                    self.edge_features,
                    self.link_pairs
                )
                link_features_seq = [link_features]
                fused_features = temporal_fusion(
                    link_features_seq,
                    gamma=self.config.fusion_gamma
                )
                y_pred = self.tfn(fused_features)
                loss = tgnn_loss(y_pred, batch_y)
                val_loss += loss.item()
        
        return val_loss / len(self.test_loader)
    
    def save_checkpoint(self, epoch):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.tgnn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'gbdt_model': self.gbdt
        }
        
        checkpoint_dir = os.path.join(self.config.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        torch.save(
            checkpoint,
            os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        )
        
    def train(self):
        """完整的训练流程"""
        try:
            self.prepare_data()
            self.train_gbdt()
            self.train_tgnn()
            self.logger.info("训练完成！")
        except Exception as e:
            self.logger.error(f"训练过程中出现错误: {str(e)}")
            raise

def main():
    # 加载配置
    config = Config()
    
    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main() 