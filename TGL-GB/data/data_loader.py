import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .feature_extractor import FeatureExtractor
from .dynamic_feature_extractor import DynamicFeatureExtractor
from models.feature_initializer import FeatureInitializer

class DataLoader:
    def __init__(self, test_size=0.2, random_state=0):
        self.test_size = test_size
        self.random_state = random_state
        self.feature_extractor = FeatureExtractor()
        self.dynamic_feature_extractor = DynamicFeatureExtractor()
        self.feature_initializer = FeatureInitializer()
        
    def _load_raw_data(self, file_prefix):
        """加载原始数据文件"""
        try:
            data1 = np.loadtxt(f'{file_prefix}方差.txt')
            data2 = np.loadtxt(f'{file_prefix}均值.txt')
            data3 = np.loadtxt(f'{file_prefix}中值.txt')
            return np.vstack((data1, data2, data3)).T
        except Exception as e:
            print(f"加载{file_prefix}数据时出错: {str(e)}")
            return np.zeros((50, 3))  # 返回零矩阵作为默认值
        
    def _create_feature_dataframe(self, data, link_type):
        """创建包含静态特征的数据框"""
        try:
            df = pd.DataFrame({
                'ip_address': [f'192.168.{i}.{j}' for i, j in zip(range(50), range(50))],
                'asn': [f'AS{i+1000}' for i in range(50)],
                'isp': [f'ISP{i%5}' for i in range(50)],
                'link_type': link_type
            })
            return df
        except Exception as e:
            print(f"创建{link_type}特征数据框时出错: {str(e)}")
            return pd.DataFrame()  # 返回空数据框作为默认值
        
    def _create_rtt_dataframe(self, data, link_type):
        """创建包含RTT数据的数据框"""
        try:
            # 生成示例RTT数据
            timestamps = pd.date_range(start='2024-01-01', periods=10, freq='H')
            hop_counts = np.arange(1, 11)
            
            rtt_data = []
            for timestamp in timestamps:
                for hop_count in hop_counts:
                    # 根据链路类型生成不同的RTT模式
                    if link_type == 'fiber':
                        base_rtt = 5 * hop_count + np.random.normal(0, 1)
                    elif link_type == 'satellite':
                        base_rtt = 100 + 2 * hop_count + np.random.normal(0, 5)
                    elif link_type == 'microwave':
                        base_rtt = 15 + 4 * hop_count + np.random.normal(0, 2)
                    else:  # mobile
                        base_rtt = 20 + 8 * hop_count + np.random.normal(0, 3)
                        
                    rtt_data.append({
                        'timestamp': timestamp,
                        'hop_count': hop_count,
                        'rtt': base_rtt,
                        'link_type': link_type
                    })
                    
            return pd.DataFrame(rtt_data)
        except Exception as e:
            print(f"创建{link_type} RTT数据框时出错: {str(e)}")
            return pd.DataFrame()
        
    def load_fiber_data(self):
        """加载光纤数据"""
        x = self._load_raw_data('光纤')
        df = self._create_feature_dataframe(x, 'fiber')
        rtt_df = self._create_rtt_dataframe(x, 'fiber')
        return x, df, rtt_df
        
    def load_satellite_data(self):
        """加载卫星数据"""
        x = self._load_raw_data('卫星')
        df = self._create_feature_dataframe(x, 'satellite')
        rtt_df = self._create_rtt_dataframe(x, 'satellite')
        return x, df, rtt_df
        
    def load_mobile_data(self):
        """加载移动数据"""
        x = self._load_raw_data('移动')
        df = self._create_feature_dataframe(x, 'mobile')
        rtt_df = self._create_rtt_dataframe(x, 'mobile')
        return x, df, rtt_df
        
    def load_microwave_data(self):
        """加载微波数据"""
        x = self._load_raw_data('微波')
        df = self._create_feature_dataframe(x, 'microwave')
        rtt_df = self._create_rtt_dataframe(x, 'microwave')
        return x, df, rtt_df
        
    def load_all_data(self):
        """加载所有数据并划分训练集和测试集"""
        try:
            # 加载原始特征数据
            x_fiber, df_fiber, rtt_fiber = self.load_fiber_data()
            x_satellite, df_satellite, rtt_satellite = self.load_satellite_data()
            x_mobile, df_mobile, rtt_mobile = self.load_mobile_data()
            x_microwave, df_microwave, rtt_microwave = self.load_microwave_data()
            
            # 打印数据形状，用于调试
            print(f"光纤数据形状: {x_fiber.shape}")
            print(f"卫星数据形状: {x_satellite.shape}")
            print(f"移动数据形状: {x_mobile.shape}")
            print(f"微波数据形状: {x_microwave.shape}")
            
            # 合并原始特征
            X_raw = np.vstack((x_fiber, x_satellite, x_mobile, x_microwave))
            print(f"合并后的原始特征形状: {X_raw.shape}")
            
            # 合并静态特征数据框
            df_all = pd.concat([df_fiber, df_satellite, df_mobile, df_microwave], ignore_index=True)
            print(f"合并后的静态特征数据框形状: {df_all.shape}")
            
            # 合并RTT数据
            rtt_all = pd.concat([rtt_fiber, rtt_satellite, rtt_mobile, rtt_microwave], ignore_index=True)
            print(f"合并后的RTT数据形状: {rtt_all.shape}")
            
            # 提取静态特征
            X_static = self.feature_extractor.fit_transform(df_all)
            print(f"提取后的静态特征形状: {X_static.shape}")
            
            # 提取动态特征
            X_dynamic = self.dynamic_feature_extractor.fit_transform(rtt_all)
            print(f"提取后的动态特征形状: {X_dynamic.shape}")
            
            # 构建头节点和尾节点的特征数据框
            head_nodes = df_all.copy()
            tail_nodes = df_all.copy()
            
            # 使用特征初始化器构建完整的特征矩阵
            X = self.feature_initializer.initialize_features(
                head_nodes=head_nodes,
                tail_nodes=tail_nodes,
                edges=rtt_all
            )
            print(f"最终特征矩阵形状: {X.shape}")
            
            # 获取特征维度信息
            feature_dims = self.feature_initializer.get_feature_dimensions()
            print("特征维度信息:", feature_dims)
            
            # 创建标签
            y = np.concatenate([
                np.zeros(50),    # 光纤标签为0
                np.ones(50),     # 卫星标签为1
                np.ones(50) * 2, # 移动标签为2
                np.ones(50) * 3  # 微波标签为3
            ])
            
            # 划分训练集和测试集
            return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            raise 