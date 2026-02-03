# -*- coding: utf-8 -*-
"""
Core Adapter - 封裝對上游核心算法的調用

這個適配器將 predict/ 目錄中的核心算法封裝為統一的接口。
當 Algorithm Engineer (CaiJianxiu) 更新核心算法時，只需修改此文件。

設計模式：
- Adapter Pattern: 適配上游核心算法
- Facade Pattern: 簡化複雜的核心算法調用
- Singleton Pattern: 全局唯一的適配器實例

維護責任：
- Fullstack Engineer (你) 負責維護此適配器
- Algorithm Engineer (CaiJianxiu) 負責維護 predict/ 目錄
"""

import sys
import threading
from pathlib import Path
from typing import Tuple, Any, List, Optional
import torch
from torch_geometric.data import Data

# 添加 predict/ 目錄到 Python 路徑（導入上游核心算法）
PROJECT_ROOT = Path(__file__).parent.parent.parent
PREDICT_DIR = PROJECT_ROOT / "predict"

# 確保 predict/ 目錄存在
if not PREDICT_DIR.exists():
    raise RuntimeError(
        f"Core algorithm directory not found: {PREDICT_DIR}\n"
        "Please ensure the 'predict/' directory exists in the project root."
    )

# 將 predict/ 添加到 Python 路徑（如果尚未添加）
predict_dir_str = str(PREDICT_DIR)
if predict_dir_str not in sys.path:
    sys.path.insert(0, predict_dir_str)

# 導入上游核心算法模塊（不是複製代碼！）
try:
    from aop_def import CombinedModel
    from aop_dataloader import aa_to_int, aa_to_smiles, mol_to_graph
    from seq_model_def import SequenceModel
    from graph_model_def import MPNN
except ImportError as e:
    raise ImportError(
        f"Failed to import core algorithm modules from {PREDICT_DIR}.\n"
        f"Error: {e}\n"
        "Please ensure the 'predict/' directory contains the required modules."
    ) from e

# 修正 xLSTM 在 CPU 環境的後端設定（上游預設為 "cpu"，但 xlstm 僅支援 "vanilla"/"cuda"）
try:
    import seq_model_def as _seq_model_def
    _backend = getattr(_seq_model_def.cfg.slstm_block.slstm, "backend", None)
    if _backend == "cpu":
        _seq_model_def.cfg.slstm_block.slstm.backend = "vanilla"
except Exception:
    # 避免影響主流程，必要時由上游或環境修正
    pass

# 導入適配器接口
from app.adapters.interfaces import IPredictorCore, IDataProcessor, IModelInfo
from app.adapters.exceptions import (
    ModelAdaptationError,
    DataAdaptationError,
    CoreImportError
)
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class CoreAdapter(IPredictorCore, IDataProcessor, IModelInfo):
    """
    核心算法適配器
    
    這個類封裝了對 predict/ 目錄中核心算法的所有調用。
    當上游算法更新時，只需要修改這個類的實現。
    
    職責：
    1. 導入並封裝上游核心算法
    2. 提供統一的接口給 FastAPI 服務層
    3. 處理核心算法的版本差異
    4. 轉換數據格式（如果需要）
    
    使用方式：
        adapter = get_core_adapter()  # 獲取單例
        seq_tensor, graph = adapter.process_sequence("ACDEFGH", 50)
    """
    
    # 支持的氨基酸列表（來自上游算法）
    SUPPORTED_AA = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    
    def __init__(self):
        """初始化適配器"""
        self._model_class = CombinedModel
        logger.info("CoreAdapter initialized successfully")
        logger.info(f"Core algorithm location: {PREDICT_DIR}")
    
    # ============================================================
    # IPredictorCore 接口實現
    # ============================================================
    
    def load_model(self, model_path: str, device: torch.device) -> torch.nn.Module:
        """
        加載模型 - 適配上游的模型加載邏輯
        
        這個方法封裝了上游 aop_predict.py 中的模型加載邏輯。
        如果上游的加載方式改變，只需修改這個方法。
        
        Args:
            model_path: 模型文件路徑
            device: 運行設備 (cpu/cuda)
        
        Returns:
            加載好的模型實例
        
        Raises:
            ModelAdaptationError: 模型加載失敗
        """
        try:
            logger.info(f"Loading model from {model_path} on device {device}")
            
            # 創建模型實例
            model = self._model_class()
            
            # 加載 checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # 處理不同的 checkpoint 格式（適配上游可能的格式變化）
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    logger.debug("Loading from checkpoint with 'model_state_dict' key")
                else:
                    state_dict = checkpoint
                    logger.debug("Loading from checkpoint dictionary")
            else:
                state_dict = checkpoint
                logger.debug("Loading from state dict directly")
            
            # 過濾形狀不匹配的權重（CPU backend 可能與 checkpoint 形狀不同）
            model_state = model.state_dict()
            filtered_state = {}
            skipped_keys = []
            for k, v in state_dict.items():
                if k in model_state and v.shape == model_state[k].shape:
                    filtered_state[k] = v
                else:
                    skipped_keys.append(k)

            if skipped_keys:
                logger.warning(
                    f"Skipped {len(skipped_keys)} keys due to shape mismatch."
                )

            # 加載權重（使用 strict=False 以支持部分加載）
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys when loading model: {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading model: {unexpected_keys[:5]}...")
            
            # 移動到設備並設置為評估模式
            model.to(device)
            model.eval()
            
            logger.info("Model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelAdaptationError(
                f"Failed to load model from {model_path}: {str(e)}",
                model_path=model_path
            ) from e
    
    def process_sequence(self, sequence: str, seq_length: int) -> Tuple[torch.Tensor, Data]:
        """
        處理序列 - 適配上游的數據處理邏輯
        
        這個方法封裝了上游 aop_dataloader.py 中的數據處理流程：
        1. 序列 → 整數編碼
        2. 序列 → SMILES → 分子圖
        
        Args:
            sequence: 氨基酸序列
            seq_length: 序列最大長度
        
        Returns:
            (序列張量, 圖數據) 元組
        
        Raises:
            DataAdaptationError: 數據處理失敗
        """
        try:
            # 1. 序列轉整數編碼
            seq_int = self.aa_to_int(sequence)
            
            # 2. 填充或截斷到指定長度
            if len(seq_int) > seq_length:
                seq_int = seq_int[:seq_length]
            else:
                seq_int = seq_int + [0] * (seq_length - len(seq_int))
            
            seq_tensor = torch.tensor(seq_int, dtype=torch.long)
            
            # 3. 序列轉圖數據
            from rdkit import Chem
            
            smiles = self.aa_to_smiles(sequence)
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is not None:
                graph = self.mol_to_graph(mol)
            else:
                # 如果 SMILES 轉換失敗，創建空圖
                logger.warning(f"Failed to convert sequence to molecule: {sequence[:20]}...")
                graph = self._create_empty_graph()
            
            return seq_tensor, graph
            
        except Exception as e:
            logger.error(f"Failed to process sequence: {str(e)}")
            raise DataAdaptationError(
                f"Failed to process sequence: {str(e)}",
                sequence=sequence
            ) from e
    
    def predict(self,
                model: torch.nn.Module,
                sequences: torch.Tensor,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch: torch.Tensor,
                device: torch.device) -> torch.Tensor:
        """
        執行預測 - 適配上游的前向傳播邏輯
        
        這個方法封裝了上游模型的前向傳播。
        CombinedModel 返回多個輸出，我們只需要最後的預測值。
        
        Args:
            model: 模型實例
            sequences: 序列張量
            x: 節點特徵
            edge_index: 邊索引
            edge_attr: 邊特徵
            batch: 批次索引
            device: 運行設備
        
        Returns:
            預測輸出張量 (概率)
        """
        with torch.no_grad():
            # 上游模型返回：(seq_features, pooled_seq, graph_features, 
            #               fused_features, last_hidden, outputs)
            # 我們只需要最後的 outputs
            _, _, _, _, _, outputs = model(
                sequences.to(device),
                x.to(device),
                edge_index.to(device),
                edge_attr.to(device),
                batch.to(device)
            )
        return outputs
    
    # ============================================================
    # IDataProcessor 接口實現
    # ============================================================
    
    def aa_to_int(self, sequence: str) -> List[int]:
        """
        直接調用上游的 aa_to_int 函數
        
        這是一個薄封裝，確保即使上游函數簽名改變，
        也只需修改這裡。
        """
        return aa_to_int(sequence)
    
    def aa_to_smiles(self, sequence: str) -> str:
        """直接調用上游的 aa_to_smiles 函數"""
        return aa_to_smiles(sequence)
    
    def mol_to_graph(self, mol: Any) -> Data:
        """直接調用上游的 mol_to_graph 函數"""
        return mol_to_graph(mol)
    
    # ============================================================
    # IModelInfo 接口實現
    # ============================================================
    
    def get_model_class(self) -> type:
        """獲取模型類（用於高級用途）"""
        return self._model_class
    
    def get_supported_aa(self) -> List[str]:
        """獲取支持的氨基酸列表"""
        return self.SUPPORTED_AA.copy()
    
    # ============================================================
    # 輔助方法
    # ============================================================
    
    def _create_empty_graph(self) -> Data:
        """
        創建空圖（用於處理無效分子）
        
        Returns:
            空的 PyTorch Geometric Data 對象
        """
        return Data(
            x=torch.zeros((1, 12), dtype=torch.float),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 3), dtype=torch.float)
        )
    
    def get_core_version(self) -> str:
        """
        獲取核心算法版本（如果上游提供）
        
        Returns:
            版本字符串，如果無法確定則返回 "unknown"
        """
        # 這是一個預留方法，如果上游提供版本信息，可以在這裡讀取
        return "unknown"


# ============================================================
# Singleton Pattern - 全局適配器實例
# ============================================================

_adapter_instance: Optional[CoreAdapter] = None
_adapter_lock = threading.Lock()


def get_core_adapter() -> CoreAdapter:
    """
    獲取核心適配器單例
    
    使用單例模式確保整個應用只有一個適配器實例。
    線程安全的雙重檢查鎖定。
    
    Returns:
        CoreAdapter 實例
    """
    global _adapter_instance
    
    if _adapter_instance is None:
        with _adapter_lock:
            # 雙重檢查鎖定
            if _adapter_instance is None:
                _adapter_instance = CoreAdapter()
                logger.info("Created singleton CoreAdapter instance")
    
    return _adapter_instance


def reset_adapter() -> None:
    """
    重置適配器單例（主要用於測試）
    
    警告：只應在測試環境中使用
    """
    global _adapter_instance
    with _adapter_lock:
        _adapter_instance = None
        logger.warning("CoreAdapter singleton has been reset")
