import numpy as np
from sklearn.preprocessing import OneHotEncoder
import ipaddress
import pandas as pd

class FeatureExtractor:
    def __init__(self):
        self.ip_encoder = None
        self.subnet_encoder = OneHotEncoder(handle_unknown='ignore')
        self.asn_encoder = None
        self.isp_encoder = OneHotEncoder(handle_unknown='ignore')
        
    def _ip_to_int(self, ip_str):
        """将IP地址转换为整数"""
        try:
            return int(ipaddress.IPv4Address(ip_str))
        except:
            return 0
            
    def _extract_subnet(self, ip_str):
        """从IP地址中提取子网信息"""
        try:
            ip = ipaddress.IPv4Network(ip_str + '/24', strict=False)
            return str(ip.network_address)
        except:
            return '0.0.0.0'
            
    def _normalize_asn(self, asn):
        """标准化ASN号码"""
        try:
            return int(str(asn).replace('AS', ''))
        except:
            return 0
            
    def fit_transform(self, data):
        """
        处理输入数据并提取特征
        
        参数:
        data: DataFrame，包含以下列：
            - ip_address: IP地址
            - asn: ASN号码
            - isp: ISP信息
        """
        # 提取IP特征
        ip_features = data['ip_address'].apply(self._ip_to_int).values.reshape(-1, 1)
        
        # 提取子网特征
        subnet_features = data['ip_address'].apply(self._extract_subnet).values.reshape(-1, 1)
        subnet_encoded = self.subnet_encoder.fit_transform(subnet_features)
        
        # 提取ASN特征
        asn_features = data['asn'].apply(self._normalize_asn).values.reshape(-1, 1)
        
        # 提取ISP特征
        isp_features = data['isp'].values.reshape(-1, 1)
        isp_encoded = self.isp_encoder.fit_transform(isp_features)
        
        # 确保所有特征都是2D数组
        features_list = []
        
        # 添加IP特征
        features_list.append(ip_features)
        
        # 添加子网特征
        if subnet_encoded.ndim == 1:
            subnet_encoded = subnet_encoded.reshape(-1, 1)
        features_list.append(subnet_encoded)
        
        # 添加ASN特征
        features_list.append(asn_features)
        
        # 添加ISP特征
        if isp_encoded.ndim == 1:
            isp_encoded = isp_encoded.reshape(-1, 1)
        features_list.append(isp_encoded)
        
        # 打印每个特征的形状，用于调试
        for i, feat in enumerate(features_list):
            print(f"特征 {i} 的形状: {feat.shape}")
        
        # 合并所有特征
        features = np.hstack(features_list)
        
        return features
        
    def transform(self, data):
        """
        使用已训练的特征提取器转换新数据
        """
        # 提取IP特征
        ip_features = data['ip_address'].apply(self._ip_to_int).values.reshape(-1, 1)
        
        # 提取子网特征
        subnet_features = data['ip_address'].apply(self._extract_subnet).values.reshape(-1, 1)
        subnet_encoded = self.subnet_encoder.transform(subnet_features)
        
        # 提取ASN特征
        asn_features = data['asn'].apply(self._normalize_asn).values.reshape(-1, 1)
        
        # 提取ISP特征
        isp_features = data['isp'].values.reshape(-1, 1)
        isp_encoded = self.isp_encoder.transform(isp_features)
        
        # 确保所有特征都是2D数组
        features_list = []
        
        # 添加IP特征
        features_list.append(ip_features)
        
        # 添加子网特征
        if subnet_encoded.ndim == 1:
            subnet_encoded = subnet_encoded.reshape(-1, 1)
        features_list.append(subnet_encoded)
        
        # 添加ASN特征
        features_list.append(asn_features)
        
        # 添加ISP特征
        if isp_encoded.ndim == 1:
            isp_encoded = isp_encoded.reshape(-1, 1)
        features_list.append(isp_encoded)
        
        # 合并所有特征
        features = np.hstack(features_list)
        
        return features 