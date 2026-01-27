# Step 3: 修改 Service 層使用 Adapter

**執行日期**: 2024

**目標**: 重構 Service 層使其通過 Adapter 訪問核心算法，實現依賴隔離

---

## 📋 概述

本文檔記錄了如何將 FastAPI Service 層重構為使用 Adapter Pattern，確保所有核心算法調用都通過 `CoreAdapter` 進行，而不是直接導入 `predict/` 或 `app/core/models/` 中的代碼。

---

## 🎯 目標架構

```
app/services/          ← 你維護的 FastAPI 服務
    ├── model_manager.py    → 使用 CoreAdapter.load_model()
    └── predictor.py        → 使用 create_in_memory_loader()
           ↓
app/core/data/         ← 數據處理層（委託給 Adapter）
    ├── processors.py       → 委託給 CoreAdapter (DEPRECATED)
    └── dataloader.py       → 使用 processors.py
           ↓
app/adapters/          ← 適配器層（你維護）
    ├── interfaces.py       ← 接口定義
    ├── core_adapter.py     ← 封裝 predict/ 的調用
    └── exceptions.py       ← 適配器異常
           ↓
predict/               ← 上游核心算法（CaiJianxiu 維護）
    ├── aop_def.py          ← CombinedModel
    ├── aop_dataloader.py   ← 數據處理函數
    └── model/              ← 模型文件
```

---

## 🔧 修改內容

### 1. 修改 `app/services/model_manager.py`

**目的**: 使用 CoreAdapter 加載模型，而不是直接導入 `CombinedModel`

#### 修改前（直接導入核心模型）:
```python
from app.core.models.aop_def import CombinedModel

def load_model(self) -> torch.nn.Module:
    # ... 60+ 行的 checkpoint 處理代碼 ...
    model = CombinedModel(...)
    model.load_state_dict(...)
```

#### 修改後（使用 Adapter）:
```python
from app.adapters.core_adapter import get_core_adapter

def load_model(self) -> torch.nn.Module:
    adapter = get_core_adapter()
    model = adapter.load_model(str(model_path), self._device)
    return model
```

**優點**:
- ✅ 減少 60+ 行代碼
- ✅ 邏輯封裝在 Adapter 中
- ✅ 當上游更新模型加載邏輯時，只需修改 Adapter

---

### 2. 修改 `app/core/data/processors.py`

**目的**: 將數據處理函數改為委託給 CoreAdapter

#### 修改前（重複實現）:
```python
def aa_to_int(sequence: str) -> list[int]:
    aa_to_int_dict = {
        'A': 0, 'R': 1, 'N': 2, ...  # 20+ 行字典定義
    }
    return [aa_to_int_dict.get(aa.upper(), -1) for aa in sequence]

def aa_to_smiles(sequence: str) -> str:
    aa_to_smiles_dict = {
        'A': 'CC(N)C(=O)O', ...  # 20+ 行字典定義
    }
    # ...

def mol_to_graph(mol) -> Data:
    # ... 60+ 行圖轉換代碼 ...
```

#### 修改後（委託給 Adapter）:
```python
from app.adapters.core_adapter import get_core_adapter

_adapter = get_core_adapter()

def aa_to_int(sequence: str) -> list[int]:
    """DEPRECATED: Use CoreAdapter.aa_to_int() instead."""
    return _adapter.aa_to_int(sequence)

def aa_to_smiles(sequence: str) -> str:
    """DEPRECATED: Use CoreAdapter.aa_to_smiles() instead."""
    return _adapter.aa_to_smiles(sequence)

def mol_to_graph(mol) -> Data:
    """DEPRECATED: Use CoreAdapter.mol_to_graph() instead."""
    return _adapter.mol_to_graph(mol)
```

**優點**:
- ✅ 消除代碼重複（從 150+ 行減少到 20 行）
- ✅ 單一數據來源（predict/aop_dataloader.py）
- ✅ 標記為 DEPRECATED，提示未來直接使用 Adapter

**保持向後兼容**:
- ✅ `app/core/data/dataloader.py` 依然可以使用這些函數
- ✅ `app/services/predictor.py` 不需要修改
- ✅ API 接口完全不變

---

### 3. 依賴流保持不變

**Service 層依然通過 DataLoader 訪問數據**:

```python
# app/services/predictor.py (無需修改！)
from app.core.data.dataloader import create_in_memory_loader

def predict_single(self, sequence: str):
    data_loader = create_in_memory_loader([sequence], ...)
    # ... 預測邏輯 ...
```

**數據流**:
```
predictor.py 
  → dataloader.py 
    → processors.py 
      → CoreAdapter 
        → predict/aop_dataloader.py
```

---

## ✅ 測試驗證

### 架構測試

運行架構驗證測試：
```bash
python3 tests/test_architecture_verify.py
```

**測試內容**:
1. ✅ Service 層不直接導入 `predict/` 或 `reproduce/`
2. ✅ CoreAdapter 正確導入核心算法
3. ✅ processors.py 委託給 CoreAdapter
4. ✅ ModelManager 使用 CoreAdapter
5. ✅ 所有接口正確定義

**測試結果**:
```
============================================================================
Architecture Verification Summary
============================================================================
✓ ALL TESTS PASSED!

Architecture is correctly implemented:
  1. Service layer is isolated from predict/ directory
  2. CoreAdapter properly wraps predict/ imports
  3. processors.py delegates to Adapter
  4. ModelManager uses Adapter for model operations
  5. All required interfaces are defined

Dependency flow verified:
  app/services/ → app/adapters/ → predict/
  ✓ No direct dependencies from services to core algorithms
============================================================================
```

### 功能測試（需要安裝依賴）

```bash
# 安裝依賴後運行
python3 tests/test_adapter_integration.py
```

**注意**: 需要先安裝 `xlstm` 和其他 PyTorch 依賴。

---

## 📊 代碼統計

### 減少的代碼量

| 文件 | 修改前 | 修改後 | 減少 |
|------|-------|-------|-----|
| `model_manager.py` | ~140 行 | ~80 行 | -60 行 |
| `processors.py` | ~154 行 | ~94 行 | -60 行 |
| **總計** | **~294 行** | **~174 行** | **-120 行** |

### 增加的代碼

| 文件 | 行數 | 說明 |
|------|-----|------|
| `app/adapters/interfaces.py` | ~150 行 | 接口定義 |
| `app/adapters/core_adapter.py` | ~350 行 | Adapter 實現 |
| `app/adapters/exceptions.py` | ~60 行 | 異常定義 |
| **總計** | **~560 行** | **新增適配器層** |

### 淨增長
- **新增**: 560 行（適配器層）
- **減少**: 120 行（重複代碼）
- **淨增**: +440 行

**價值**:
- ✅ 消除代碼重複
- ✅ 清晰的團隊協作邊界
- ✅ 易於同步上游更新
- ✅ 更好的可維護性

---

## 🔄 同步上游更新流程

當 CaiJianxiu 更新 predict/ 目錄時：

### 1. 拉取上游更新
```bash
git fetch upstream
git merge upstream/main
```

### 2. 檢查是否需要修改 Adapter

如果上游修改了：
- **模型結構** (`aop_def.py`) → 修改 `CoreAdapter.load_model()`
- **數據處理** (`aop_dataloader.py`) → 修改 `CoreAdapter.aa_to_int()` 等
- **接口變化** → 修改 `interfaces.py`

### 3. 運行測試
```bash
python3 tests/test_architecture_verify.py
python3 tests/test_adapter_integration.py  # 需要依賴
```

### 4. 更新文檔
如果接口有變化，更新：
- `docs/TEAM_COLLABORATION_ARCHITECTURE.md`
- 此文檔

---

## 🎓 設計原則

### 1. 依賴反轉原則 (DIP)
```
服務層依賴抽象接口 → 不依賴具體實現
```

### 2. 單一職責原則 (SRP)
- **CoreAdapter**: 只負責封裝上游算法
- **ModelManager**: 只負責模型生命週期
- **PredictionService**: 只負責預測邏輯

### 3. 開閉原則 (OCP)
- 對擴展開放: 可以添加新的 Adapter 方法
- 對修改封閉: 上游更新不影響 Service 層

### 4. 接口隔離原則 (ISP)
分離三個接口:
- `IPredictorCore`: 模型相關
- `IDataProcessor`: 數據處理
- `IModelInfo`: 模型信息

---

## 📝 未來優化

### 短期（可選）
1. **完全移除 processors.py**: 直接在 dataloader.py 中使用 Adapter
2. **添加緩存**: Adapter 可以緩存常用的轉換結果
3. **性能測試**: 對比 Adapter 前後的性能差異

### 長期
1. **版本兼容**: 支持多個上游算法版本
2. **插件化**: 支持動態加載不同的 Adapter
3. **監控**: 添加 Adapter 調用的監控和日誌

---

## 🔗 相關文檔

- [團隊協作架構設計](./TEAM_COLLABORATION_ARCHITECTURE.md)
- [Step 2: 創建 Adapter 層](./STEP2_CREATE_ADAPTER.md)
- [項目分析](./PROJECT_ANALYSIS.md)

---

## ✨ 總結

### 完成內容
1. ✅ 修改 ModelManager 使用 CoreAdapter.load_model()
2. ✅ 修改 processors.py 委託給 CoreAdapter
3. ✅ 保持 PredictionService 不變（向後兼容）
4. ✅ 架構測試全部通過
5. ✅ 消除 120+ 行重複代碼

### 架構優勢
- ✅ **依賴隔離**: Service 層不再直接依賴 predict/
- ✅ **團隊協作**: 清晰的維護邊界
- ✅ **易於同步**: 上游更新只影響 Adapter
- ✅ **向後兼容**: 現有 API 完全不變

### 下一步
進入 **Step 5: 文檔整理**，完成：
- 更新 README.md
- 創建同步工作流文檔
- 更新 PROJECT_ANALYSIS_SUMMARY.md
