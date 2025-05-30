import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import pandas as pd

class DynamicFeatureExtractor:
    def __init__(self, polynomial_degree=3):
        """
        初始化动态特征提取器
        
        参数:
        polynomial_degree: 多项式拟合的阶数
        """
        self.polynomial_degree = polynomial_degree
        self.scaler = StandardScaler()
        self.coefficient_scaler = StandardScaler()
        
    def _polynomial_fit(self, x, y):
        """
        使用多项式拟合RTT数据
        
        参数:
        x: 跳数数据
        y: RTT数据
        
        返回:
        coefficients: 多项式系数
        """
        def objective_function(coeffs, x, y):
            """最小化拟合误差的目标函数"""
            y_pred = np.polyval(coeffs, x)
            return np.sum((y - y_pred) ** 2)
            
        # 初始系数猜测
        initial_coeffs = np.ones(self.polynomial_degree + 1)
        
        # 最小化目标函数
        result = minimize(
            objective_function,
            initial_coeffs,
            args=(x, y),
            method='Nelder-Mead'
        )
        
        return result.x
        
    def _extract_rtt_features(self, rtt_data):
        """
        从RTT数据中提取特征
        
        参数:
        rtt_data: DataFrame，包含以下列：
            - hop_count: 跳数
            - rtt: RTT值
            - timestamp: 时间戳
            
        返回:
        features: 提取的特征
        """
        # 计算最小RTT和RTT方差
        min_rtt = rtt_data.groupby('timestamp')['rtt'].min()
        var_rtt = rtt_data.groupby('timestamp')['rtt'].var()
        
        # 对每个时间戳进行多项式拟合
        coefficients_list = []
        for timestamp in rtt_data['timestamp'].unique():
            time_data = rtt_data[rtt_data['timestamp'] == timestamp]
            coeffs = self._polynomial_fit(time_data['hop_count'], time_data['rtt'])
            coefficients_list.append(coeffs)
            
        # 将系数转换为numpy数组
        coefficients_array = np.array(coefficients_list)
        
        # 标准化系数
        normalized_coefficients = self.coefficient_scaler.fit_transform(coefficients_array)
        
        # 计算每个时间戳的动态特征
        dynamic_features = []
        for i, timestamp in enumerate(rtt_data['timestamp'].unique()):
            features = {
                'timestamp': timestamp,
                'min_rtt': min_rtt[timestamp],
                'var_rtt': var_rtt[timestamp],
                'coefficients': normalized_coefficients[i]
            }
            dynamic_features.append(features)
            
        return pd.DataFrame(dynamic_features)
        
    def fit_transform(self, rtt_data):
        """
        处理输入数据并提取动态特征
        
        参数:
        rtt_data: DataFrame，包含以下列：
            - hop_count: 跳数
            - rtt: RTT值
            - timestamp: 时间戳
            
        返回:
        features: 提取的动态特征
        """
        # 提取RTT特征
        rtt_features = self._extract_rtt_features(rtt_data)
        
        # 标准化最小RTT和RTT方差
        rtt_features[['min_rtt', 'var_rtt']] = self.scaler.fit_transform(
            rtt_features[['min_rtt', 'var_rtt']]
        )
        
        return rtt_features
        
    def transform(self, rtt_data):
        """
        使用已训练的特征提取器转换新数据
        
        参数:
        rtt_data: DataFrame，包含以下列：
            - hop_count: 跳数
            - rtt: RTT值
            - timestamp: 时间戳
            
        返回:
        features: 提取的动态特征
        """
        # 提取RTT特征
        rtt_features = self._extract_rtt_features(rtt_data)
        
        # 使用已训练的scaler转换最小RTT和RTT方差
        rtt_features[['min_rtt', 'var_rtt']] = self.scaler.transform(
            rtt_features[['min_rtt', 'var_rtt']]
        )
        
        return rtt_features 