# -*- coding: utf-8 -*-
"""
接口契約定義 - 定義 FastAPI 需要的最小接口

這是 Fullstack Engineer 和 Algorithm Engineer 之間的「契約」
當上游算法更新時，只要滿足這些接口，就不會影響 FastAPI 服務
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict
import torch
from torch_geometric.data import Data


class IPredictorCore(ABC):
    """
    預測核心接口 - 定義核心算法必須提供的能力
    
    這個接口定義了 FastAPI 服務層需要的最小功能集
    """
    
    @abstractmethod
    def load_model(self, model_path: str, device: torch.device) -> torch.nn.Module:
        """
        加載模型
        
        Args:
            model_path: 模型文件路徑
            device: 運行設備 (cpu/cuda)
        
        Returns:
            加載好的模型實例
        """
        pass
    
    @abstractmethod
    def process_sequence(self, sequence: str, seq_length: int) -> Tuple[torch.Tensor, Data]:
        """
        處理單個序列數據
        
        Args:
            sequence: 氨基酸序列字符串
            seq_length: 序列最大長度
        
        Returns:
            (序列張量, 圖數據) 元組
        """
        pass
    
    @abstractmethod
    def predict(self, 
                model: torch.nn.Module,
                sequences: torch.Tensor,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor,
                device: torch.device) -> torch.Tensor:
        """
        執行預測
        
        Args:
            model: 模型實例
            sequences: 序列張量
            x: 節點特徵
            edge_index: 邊索引
            edge_attr: 邊特徵
            batch: 批次索引
            device: 運行設備
        
        Returns:
            預測輸出張量
        """
        pass


class IDataProcessor(ABC):
    """
    數據處理接口 - 定義數據轉換功能
    """
    
    @abstractmethod
    def aa_to_int(self, sequence: str) -> List[int]:
        """
        氨基酸序列轉整數編碼
        
        Args:
            sequence: 氨基酸序列 (如 "ACDEFGH")
        
        Returns:
            整數編碼列表
        """
        pass
    
    @abstractmethod
    def aa_to_smiles(self, sequence: str) -> str:
        """
        氨基酸序列轉 SMILES 表示
        
        Args:
            sequence: 氨基酸序列
        
        Returns:
            SMILES 字符串
        """
        pass
    
    @abstractmethod
    def mol_to_graph(self, mol: Any) -> Data:
        """
        分子轉圖數據結構
        
        Args:
            mol: RDKit 分子對象
        
        Returns:
            PyTorch Geometric Data 對象
        """
        pass


class IModelInfo(ABC):
    """
    模型信息接口 - 獲取模型元數據
    """
    
    @abstractmethod
    def get_model_class(self) -> type:
        """
        獲取模型類
        
        Returns:
            模型類（用於高級用途）
        """
        pass
    
    @abstractmethod
    def get_supported_aa(self) -> List[str]:
        """
        獲取支持的氨基酸列表
        
        Returns:
            氨基酸字符列表
        """
        pass
