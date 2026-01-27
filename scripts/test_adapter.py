#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試 CoreAdapter 是否正常工作

這個腳本驗證：
1. 能否成功導入上游核心算法
2. 能否創建 Adapter 實例
3. 能否處理序列數據
4. 能否加載模型（如果模型文件存在）

使用方式：
    python scripts/test_adapter.py
"""

import sys
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.adapters.core_adapter import get_core_adapter
from app.utils.logging_config import setup_logging, get_logger

# 設置日誌
setup_logging()
logger = get_logger(__name__)


def test_adapter_creation():
    """測試適配器創建"""
    print("\n" + "="*60)
    print("測試 1: 創建 CoreAdapter 實例")
    print("="*60)
    
    try:
        adapter = get_core_adapter()
        print(f"✅ 成功創建 CoreAdapter 實例")
        print(f"   支持的氨基酸: {', '.join(adapter.get_supported_aa()[:10])}...")
        return adapter
    except Exception as e:
        print(f"❌ 創建 CoreAdapter 失敗: {e}")
        return None


def test_sequence_processing(adapter):
    """測試序列處理"""
    print("\n" + "="*60)
    print("測試 2: 處理序列數據")
    print("="*60)
    
    test_sequences = [
        "ACDEFGH",
        "MKLLVVVFCLVLAAP",
        "ARNDCEQGHILKMFPSTWYV"
    ]
    
    for seq in test_sequences:
        try:
            seq_tensor, graph = adapter.process_sequence(seq, seq_length=50)
            print(f"✅ 序列 '{seq}' 處理成功")
            print(f"   序列張量形狀: {seq_tensor.shape}")
            print(f"   圖節點數: {graph.x.shape[0]}, 邊數: {graph.edge_index.shape[1]}")
        except Exception as e:
            print(f"❌ 序列 '{seq}' 處理失敗: {e}")


def test_model_loading(adapter):
    """測試模型加載"""
    print("\n" + "="*60)
    print("測試 3: 加載模型")
    print("="*60)
    
    import torch
    from app.config import settings
    
    try:
        model_path = settings.get_model_path()
        if not model_path.exists():
            print(f"⚠️  模型文件不存在: {model_path}")
            print(f"   跳過模型加載測試")
            return None
        
        device = torch.device(settings.DEVICE)
        model = adapter.load_model(str(model_path), device)
        print(f"✅ 成功加載模型")
        print(f"   模型路徑: {model_path}")
        print(f"   設備: {device}")
        print(f"   模型類型: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"❌ 模型加載失敗: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_prediction(adapter, model):
    """測試預測"""
    if model is None:
        print("\n⚠️  模型未加載，跳過預測測試")
        return
    
    print("\n" + "="*60)
    print("測試 4: 執行預測")
    print("="*60)
    
    import torch
    from app.config import settings
    
    test_sequence = "ACDEFGHIKLMNPQRSTVWY"
    
    try:
        # 處理序列
        seq_tensor, graph = adapter.process_sequence(test_sequence, seq_length=50)
        
        # 準備批次數據
        sequences = seq_tensor.unsqueeze(0)  # [1, seq_length]
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        batch = torch.zeros(x.shape[0], dtype=torch.long)
        
        device = torch.device(settings.DEVICE)
        
        # 執行預測
        outputs = adapter.predict(
            model, sequences, x, edge_index, edge_attr, batch, device
        )
        
        probability = outputs.squeeze().cpu().item()
        prediction = 1 if probability > 0.5 else 0
        
        print(f"✅ 預測成功")
        print(f"   序列: {test_sequence}")
        print(f"   預測概率: {probability:.4f}")
        print(f"   預測結果: {'AOP (抗氧化肽)' if prediction == 1 else '非 AOP'}")
        
    except Exception as e:
        print(f"❌ 預測失敗: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主測試流程"""
    print("\n" + "="*60)
    print("CoreAdapter 測試腳本")
    print("="*60)
    print("這個腳本驗證 Adapter 層是否正確封裝了上游核心算法")
    
    # 測試 1: 創建適配器
    adapter = test_adapter_creation()
    if adapter is None:
        print("\n❌ 適配器創建失敗，無法繼續測試")
        return 1
    
    # 測試 2: 序列處理
    test_sequence_processing(adapter)
    
    # 測試 3: 模型加載
    model = test_model_loading(adapter)
    
    # 測試 4: 預測
    test_prediction(adapter, model)
    
    print("\n" + "="*60)
    print("測試完成！")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
