import torch
import numpy as np
from models.gbdt_model import GBDTModel
from models.tgnn import TGNNModel, build_link_features, TemporalFusionNetwork, temporal_fusion, tgnn_loss
from data.data_loader import DataLoader

def test_model():
    print("开始测试模型...")
    
    # 1. 加载数据
    print("\n1. 加载数据")
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.load_all_data()
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 2. 测试GBDT模型
    print("\n2. 测试GBDT模型")
    gbdt = GBDTModel(
        n_trees=10,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        embedding_dim=64
    )
    print("训练GBDT模型...")
    gbdt.fit(X_train, y_train)
    
    # 预测并获取嵌入
    predictions, gbdt_embedded = gbdt.predict(X_test)
    print(f"GBDT预测结果形状: {predictions.shape}")
    print(f"GBDT嵌入特征形状: {gbdt_embedded.shape}")
    
    # 3. 测试T-GNN模型
    print("\n3. 测试T-GNN模型")
    input_dim = X_train.shape[1] + gbdt_embedded.shape[1]
    tgnn = TGNNModel(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=64,
        num_classes=4  # 4种链路类型：光纤、卫星、移动、微波
    )
    
    # 准备T-GNN输入
    X_enhanced = np.hstack([X_train, gbdt_embedded.detach().numpy()])
    X_enhanced_tensor = torch.tensor(X_enhanced, dtype=torch.float32)
    
    # 模拟边索引和时间步
    num_nodes = X_train.shape[0]
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)  # 示例边
    time_steps = torch.arange(num_nodes, dtype=torch.float32)
    
    print("训练T-GNN模型...")
    logits, node_final_emb = tgnn(X_enhanced_tensor, edge_index, time_steps)
    print(f"T-GNN输出logits形状: {logits.shape}")
    print(f"节点最终嵌入形状: {node_final_emb.shape}")
    
    # 4. 测试链路特征构建
    print("\n4. 测试链路特征构建")
    # 模拟边特征和链路对
    edge_features = torch.randn(edge_index.shape[1], 10)  # 示例边特征
    link_pairs = [(0, 1), (1, 2), (2, 0)]  # 示例链路对
    
    link_features = build_link_features(node_final_emb, edge_features, link_pairs)
    print(f"链路特征形状: {link_features.shape}")
    
    # 5. 测试时间融合
    print("\n5. 测试时间融合")
    link_features_seq = [link_features] * 3  # 模拟3个时间步
    fused_link_features = temporal_fusion(link_features_seq, gamma=0.9)
    print(f"融合后的链路特征形状: {fused_link_features.shape}")
    
    # 6. 测试TFN分类
    print("\n6. 测试TFN分类")
    tfn = TemporalFusionNetwork(
        input_dim=fused_link_features.shape[1],
        hidden_dim=64,
        num_classes=4  # 4种链路类型
    )
    y_pred = tfn(fused_link_features)
    print(f"TFN预测结果形状: {y_pred.shape}")
    
    # 7. 计算损失
    print("\n7. 计算损失")
    # 模拟标签
    labels = torch.randint(0, 4, (len(link_pairs),), dtype=torch.long)
    loss = tgnn_loss(y_pred, labels)
    print(f"模型损失: {loss.item():.4f}")
    
    # 8. 打印模型参数统计
    print("\n8. 模型参数统计")
    total_params = sum(p.numel() for p in tgnn.parameters())
    print(f"T-GNN模型总参数量: {total_params:,}")
    
    # 9. 打印分类结果
    print("\n9. 分类结果")
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / len(labels)
    print(f"准确率: {accuracy:.4f}")
    
    # 10. 打印各类别预测分布
    print("\n10. 预测分布")
    for i in range(4):
        count = (predicted == i).sum().item()
        print(f"类别 {i} (光纤/卫星/移动/微波): {count} 个样本")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_model() 