# 團隊協作架構實施指南

## 📋 Step 2 完成確認

### ✅ 已完成的工作

#### 1. Adapter 層架構創建

```
app/adapters/
├── __init__.py           ✅ 已創建 - Package 初始化
├── interfaces.py         ✅ 已創建 - 接口契約定義
├── exceptions.py         ✅ 已創建 - 適配器異常
└── core_adapter.py       ✅ 已創建 - 核心適配器實現
```

#### 2. 測試驗證

```bash
# Adapter 結構測試
✅ 能夠正確導入 predict/ 目錄的模塊
✅ 接口定義完整（IPredictorCore, IDataProcessor, IModelInfo）
✅ 單例模式實現正確（get_core_adapter）
✅ 異常處理完善

# 待安裝依賴後可完整測試
⏳ 序列處理測試（需要 xlstm, rdkit 等依賴）
⏳ 模型加載測試
⏳ 預測功能測試
```

---

## 🎯 架構設計說明

### 三層隔離架構

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Upstream Core (Algorithm Engineer 維護)            │
│  predict/                                                     │
│  ├── aop_def.py          ← CombinedModel                    │
│  ├── aop_dataloader.py   ← 數據處理函數                      │
│  ├── seq_model_def.py    ← xLSTM 序列模型                   │
│  └── graph_model_def.py  ← MPNN 圖模型                      │
│                                                               │
│  特點：只通過 git upstream 同步，不修改                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ import (導入，不複製)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Core Adapter (Fullstack Engineer 維護 - 你)        │
│  app/adapters/                                                │
│  ├── interfaces.py       ← 接口契約                          │
│  ├── core_adapter.py     ← 適配器實現                        │
│  └── exceptions.py       ← 異常處理                          │
│                                                               │
│  特點：封裝上游算法，提供統一接口                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 依賴注入
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Application (Fullstack Engineer 維護 - 你)         │
│  app/services/                                                │
│  ├── predictor.py        ← 調用 Adapter                      │
│  └── model_manager.py    ← 調用 Adapter                      │
│                                                               │
│  app/api/                                                     │
│  └── v1/routes.py        ← REST API 端點                     │
│                                                               │
│  特點：業務邏輯層，不直接依賴核心算法                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 關鍵設計模式

### 1. Adapter Pattern (適配器模式)

**目的**：將上游核心算法的接口適配為 FastAPI 需要的接口

**實現**：
- `CoreAdapter` 類實現了 `IPredictorCore`, `IDataProcessor`, `IModelInfo` 接口
- 封裝對 `predict/` 模塊的所有調用
- 當上游接口變化時，只需修改 Adapter

### 2. Facade Pattern (門面模式)

**目的**：簡化複雜的核心算法調用

**實現**：
- `process_sequence()` 封裝了複雜的數據處理流程
- `load_model()` 封裝了模型加載的各種情況
- Service 層不需要知道底層細節

### 3. Dependency Injection (依賴注入)

**目的**：便於測試和模擬

**實現**：
- Service 層接受 `CoreAdapter` 作為可選參數
- 可以注入 Mock Adapter 進行單元測試

### 4. Singleton Pattern (單例模式)

**目的**：全局唯一的適配器實例

**實現**：
- `get_core_adapter()` 函數使用線程安全的雙重檢查鎖定
- 避免重複初始化

---

## 🔄 上游更新同步流程

### 當 Algorithm Engineer 更新核心算法時

```bash
# Step 1: 獲取上游更新
git fetch upstream

# Step 2: 查看變更
git log HEAD..upstream/main --oneline
git diff HEAD..upstream/main predict/

# Step 3: 合併上游（只合併 predict/ 目錄）
git checkout development
git merge upstream/main

# Step 4: 解決衝突（如果有）
# predict/ 目錄: 接受上游版本 (Accept Upstream)
# app/ 目錄: 保留你的版本 (Keep Yours)

# Step 5: 測試 Adapter 是否需要調整
python3 test_adapter_simple.py

# Step 6: 如果接口變化，更新 Adapter
# 編輯 app/adapters/core_adapter.py

# Step 7: 提交
git commit -m "chore: sync with upstream algorithm v1.2.0"
git push origin development
```

---

## 📊 接口契約說明

### IPredictorCore - 預測核心接口

| 方法 | 用途 | 上游來源 |
|------|------|---------|
| `load_model()` | 加載模型 | `aop_predict.py` 的模型加載邏輯 |
| `process_sequence()` | 處理序列 | `aop_dataloader.py` 的數據處理 |
| `predict()` | 執行預測 | `CombinedModel` 的前向傳播 |

### IDataProcessor - 數據處理接口

| 方法 | 用途 | 上游來源 |
|------|------|---------|
| `aa_to_int()` | 序列→整數 | `aop_dataloader.aa_to_int()` |
| `aa_to_smiles()` | 序列→SMILES | `aop_dataloader.aa_to_smiles()` |
| `mol_to_graph()` | 分子→圖 | `aop_dataloader.mol_to_graph()` |

---

## 🎓 最佳實踐

### DO ✅

1. **保持 predict/ 目錄原始**
   - 只通過 git upstream 同步
   - 不要在 predict/ 中添加自己的代碼

2. **所有改動在 Adapter 層**
   - 如果需要適配上游變化，修改 `core_adapter.py`
   - 保持接口契約穩定

3. **記錄上游版本**
   - 在 commit message 中記錄上游版本
   - 創建 git tag 標記同步點

### DON'T ❌

1. **不要修改 predict/ 目錄**
   - 這會導致無法同步上游更新
   - 如需修改，在 Adapter 層適配

2. **不要複製代碼**
   - 不要把 predict/ 的代碼複製到 app/core/
   - 使用導入而非複製

3. **不要跳過測試**
   - 每次同步後都要測試 Adapter
   - 確保 API 功能正常

---

## 📁 文件說明

### app/adapters/interfaces.py

定義接口契約 - 這是你和 Algorithm Engineer 之間的「協議」

```python
# 定義了 3 個接口：
- IPredictorCore: 預測相關功能
- IDataProcessor: 數據處理功能  
- IModelInfo: 模型信息功能
```

### app/adapters/core_adapter.py

核心適配器實現 - 封裝對 predict/ 的所有調用

```python
# 關鍵方法：
- load_model(): 加載模型
- process_sequence(): 處理序列
- predict(): 執行預測
- get_core_adapter(): 獲取單例
```

### app/adapters/exceptions.py

異常定義 - 便於錯誤追蹤

```python
# 定義的異常：
- AdapterError: 基礎異常
- CoreImportError: 導入失敗
- ModelAdaptationError: 模型適配失敗
- DataAdaptationError: 數據適配失敗
```

---

## 🚀 下一步驟 (Step 3)

### 修改 Service 層使用 Adapter

需要修改以下文件：
1. `app/services/predictor.py` - 使用 Adapter 而非重寫的代碼
2. `app/services/model_manager.py` - 使用 Adapter 加載模型
3. `app/core/` - 可以刪除重複的代碼

詳見：`docs/STEP3_MODIFY_SERVICES.md`（待創建）

---

## 💡 常見問題

### Q: 為什麼不用 Git Submodule？

A: Submodule 適合獨立的第三方庫，但這裡核心算法和 FastAPI 緊密協作，需要頻繁同步，Adapter 模式更靈活。

### Q: 如果上游接口大幅變化怎麼辦？

A: 
1. 先在 Adapter 中適配新接口
2. 保持對外接口不變
3. 如果必須變化，更新接口版本（IPredictorCoreV2）

### Q: 能否直接在 Service 層導入 predict/ 模塊？

A: 不建議。通過 Adapter 層隔離可以：
- 更好地處理版本變化
- 便於單元測試（可 mock Adapter）
- 清晰的責任分離

---

## 📞 聯繫

- Fullstack Engineer (你): 負責 app/ 和 Adapter 層
- Algorithm Engineer (CaiJianxiu): 負責 predict/ 核心算法

**分工明確，職責清晰！** 🎯
