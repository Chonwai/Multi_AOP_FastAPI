# 上游算法同步工作流

**版本**: 1.0  
**日期**: 2024  
**維護**: Fullstack Engineer

---

## 📋 概述

本文檔說明當 Algorithm Engineer (CaiJianxiu) 更新上游核心算法時，Fullstack Engineer 如何同步這些更新到 FastAPI fork。

---

## 🎯 架構回顧

```
┌─────────────────────────────────────────────────────────┐
│           上游倉庫 (CaiJianxiu/Multi-AOP)                │
│                                                          │
│  predict/                ← Algorithm Engineer 維護       │
│    ├── aop_def.py                                       │
│    ├── aop_dataloader.py                                │
│    ├── seq_model_def.py                                 │
│    └── graph_model_def.py                               │
└─────────────────────────────────────────────────────────┘
                           ↓ git pull upstream/main
┌─────────────────────────────────────────────────────────┐
│        你的 Fork (你的倉庫/Multi_AOP_FastAPI)             │
│                                                          │
│  app/adapters/           ← Fullstack Engineer 維護       │
│    ├── core_adapter.py   封裝對 predict/ 的調用          │
│    └── interfaces.py     定義接口契約                    │
│                          ↑                              │
│  app/services/           只通過 Adapter 訪問              │
│    ├── model_manager.py                                 │
│    └── predictor.py                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 🔄 同步流程

### Step 1: 準備工作（已完成）

```bash
# 添加上游遠程倉庫（只需執行一次）
git remote add upstream https://github.com/CaiJianxiu/Multi-AOP.git

# 驗證遠程倉庫
git remote -v
```

**輸出應該顯示**:
```
origin    https://github.com/你的用戶名/Multi_AOP_FastAPI.git (fetch)
origin    https://github.com/你的用戶名/Multi_AOP_FastAPI.git (push)
upstream  https://github.com/CaiJianxiu/Multi-AOP.git (fetch)
upstream  https://github.com/CaiJianxiu/Multi-AOP.git (push)
```

---

### Step 2: 拉取上游更新

```bash
# 1. 確保本地工作區乾淨
git status

# 2. 如果有未提交的修改，先提交或暫存
git add .
git commit -m "chore: save work before upstream sync"

# 3. 切換到主分支
git checkout main  # 或 master

# 4. 拉取上游更新
git fetch upstream

# 5. 查看上游的修改
git log upstream/main --oneline -10

# 6. 合併上游更新
git merge upstream/main
```

---

### Step 3: 檢查需要適配的變更

#### 3.1 查看 predict/ 目錄的變更

```bash
# 查看 predict/ 目錄的修改
git diff HEAD@{1} HEAD -- predict/

# 或使用 VS Code 的 Git 差異視圖
code -d HEAD@{1}:predict/aop_def.py predict/aop_def.py
```

#### 3.2 需要關注的文件

| 上游文件 | 影響的 Adapter 方法 | 檢查項目 |
|---------|-------------------|---------|
| `predict/aop_def.py` | `load_model()` | 模型結構是否變化？ |
| `predict/aop_dataloader.py` | `aa_to_int()`, `aa_to_smiles()`, `mol_to_graph()` | 數據處理邏輯是否變化？ |
| `predict/seq_model_def.py` | `get_model_class()` | 序列模型是否更新？ |
| `predict/graph_model_def.py` | `get_model_class()` | 圖模型是否更新？ |

---

### Step 4: 更新 Adapter（如需要）

#### 場景 A: 模型結構變化

**示例**: CombinedModel 添加了新的參數

```python
# predict/aop_def.py (上游更新)
class CombinedModel(nn.Module):
    def __init__(self, ..., new_param=None):  # 新參數
        # ...
```

**你需要修改**:
```python
# app/adapters/core_adapter.py
def load_model(self, model_path: str, device: torch.device):
    # 檢查上游模型加載邏輯
    # 如果有新參數，可能需要調整加載代碼
    checkpoint = torch.load(model_path, map_location=device)
    
    # 處理新的 checkpoint 格式
    if 'new_param' in checkpoint:
        # 處理新參數
        pass
    
    model = CombinedModel(...)
    return model
```

#### 場景 B: 數據處理邏輯變化

**示例**: aa_to_int 支持更多氨基酸

```python
# predict/aop_dataloader.py (上游更新)
aa_to_int = {
    'A': 0, ..., 'V': 19,
    'X': 20  # 新增未知氨基酸
}
```

**你需要修改**:
```python
# app/adapters/core_adapter.py
SUPPORTED_AA = ['A', 'R', ..., 'V', 'X']  # 更新支持列表

def aa_to_int(self, sequence: str) -> List[int]:
    # Adapter 直接調用上游函數，通常不需要修改
    # 但可能需要更新文檔
    return aa_to_int(sequence)
```

#### 場景 C: 接口簽名變化

**示例**: process_sequence 返回值變化

```python
# 上游從返回 (tensor, graph) 改為返回 dict
def process_sequence(seq, length):
    return {
        'sequence': tensor,
        'graph': graph,
        'metadata': {...}  # 新增
    }
```

**你需要修改**:
```python
# app/adapters/interfaces.py
class IDataProcessor(ABC):
    @abstractmethod
    def process_sequence(self, sequence: str, seq_length: int) -> dict:  # 改為 dict
        """Process sequence, returns dict with sequence, graph, metadata"""
        pass

# app/adapters/core_adapter.py
def process_sequence(self, sequence: str, seq_length: int) -> dict:
    return process_sequence(sequence, seq_length)

# app/core/data/processors.py
def process_sequence(sequence: str, seq_length: int = 50):
    result = _adapter.process_sequence(sequence, seq_length)
    # 可能需要轉換格式以保持向後兼容
    return result['sequence'], result['graph']
```

---

### Step 5: 運行測試

#### 5.1 架構測試（必須通過）

```bash
python3 tests/test_architecture_verify.py
```

**預期輸出**:
```
✓ ALL TESTS PASSED!
```

#### 5.2 Adapter 單元測試

```bash
python3 tests/test_adapter_simple.py
```

#### 5.3 集成測試

```bash
python3 tests/test_adapter_integration.py
```

#### 5.4 API 測試（如果可能）

```bash
# 啟動服務
uvicorn app.main:app --reload

# 測試預測 API
curl -X POST "http://localhost:8000/api/v1/predict/single" \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ACDEFGHIKLMNPQRSTVWY"}'
```

---

### Step 6: 提交修改

```bash
# 1. 查看修改
git status
git diff

# 2. 暫存修改（Adapter 相關文件）
git add app/adapters/
git add tests/

# 3. 提交修改
git commit -m "feat: sync upstream algorithm updates

- Merge upstream/main (commit: abc123)
- Update CoreAdapter to support new model format
- Add support for unknown amino acid 'X'
- All tests passing

Upstream changes:
- predict/aop_def.py: Add new parameter to CombinedModel
- predict/aop_dataloader.py: Support unknown amino acid"

# 4. 推送到你的 fork
git push origin main
```

---

## 🚨 常見問題與解決

### Q1: 合併衝突（Merge Conflicts）

**問題**: `git merge upstream/main` 出現衝突

```
CONFLICT (content): Merge conflict in predict/aop_def.py
```

**解決**:
```bash
# 1. 查看衝突文件
git status

# 2. 手動解決衝突
#    - 如果是 predict/ 目錄：使用上游版本（他們維護）
#    - 如果是 app/ 目錄：保留你的版本（你維護）

# 對於 predict/ 衝突，接受上游版本
git checkout --theirs predict/aop_def.py

# 3. 標記衝突已解決
git add predict/aop_def.py

# 4. 完成合併
git commit -m "merge: resolve conflicts with upstream"
```

---

### Q2: Adapter 導入失敗

**問題**: 
```python
ImportError: cannot import name 'CombinedModel' from 'aop_def'
```

**原因**: 上游重命名了類或函數

**解決**:
```python
# app/adapters/core_adapter.py

# 舊版本
from aop_def import CombinedModel

# 如果上游改名為 AOPModel
from aop_def import AOPModel as CombinedModel  # 使用別名保持兼容
```

---

### Q3: 測試失敗

**問題**: `test_architecture_verify.py` 測試失敗

**診斷**:
```bash
# 檢查 import 語句
python3 -c "from app.adapters.core_adapter import get_core_adapter; print('OK')"

# 檢查 predict/ 目錄
ls -la predict/

# 查看 Python 路徑
python3 -c "import sys; print('\n'.join(sys.path))"
```

**解決**: 確保 `predict/` 目錄存在且包含所有必需文件

---

### Q4: 上游依賴變化

**問題**: 上游添加了新的依賴包

**解決**:
```bash
# 1. 查看上游 requirements.txt 的變化
git diff HEAD@{1} HEAD -- requirements.txt

# 2. 更新你的 requirements.txt
# 將新依賴添加到你的 requirements.txt

# 3. 安裝新依賴
pip install -r requirements.txt

# 4. 測試
python3 tests/test_adapter_integration.py
```

---

## 📊 同步檢查清單

在合併上游更新後，使用此清單確保一切正常：

- [ ] **拉取**: `git fetch upstream && git merge upstream/main`
- [ ] **衝突**: 解決所有合併衝突（predict/ 使用上游版本）
- [ ] **依賴**: 檢查 `requirements.txt` 是否有新依賴
- [ ] **Adapter**: 檢查 `predict/` 的變更，更新 `core_adapter.py`
- [ ] **接口**: 如果接口變化，更新 `interfaces.py`
- [ ] **測試-架構**: `python3 tests/test_architecture_verify.py` ✓
- [ ] **測試-單元**: `python3 tests/test_adapter_simple.py` ✓
- [ ] **測試-集成**: `python3 tests/test_adapter_integration.py` ✓
- [ ] **文檔**: 更新 `STEP3_MODIFY_SERVICES.md` 或其他相關文檔
- [ ] **提交**: 編寫清晰的 commit message
- [ ] **推送**: `git push origin main`

---

## 🔗 相關文檔

- [Step 2: 創建 Adapter 層](./STEP2_CREATE_ADAPTER.md)
- [Step 3: 修改 Service 層](./STEP3_MODIFY_SERVICES.md)
- [團隊協作架構](./TEAM_COLLABORATION_ARCHITECTURE.md)
- [項目分析](./PROJECT_ANALYSIS.md)

---

## 💡 最佳實踐

### 1. 定期同步
- **頻率**: 每週或每次上游有重大更新時
- **時機**: 在開始新功能開發之前

### 2. 小步快跑
- 每次只合併少量 commits
- 立即測試，不要累積太多變更

### 3. 保持溝通
- 與 Algorithm Engineer 保持溝通
- 提前知道重大變更，預留適配時間

### 4. 版本標記
```bash
# 在重大同步後打標籤
git tag -a v1.1.0-upstream-sync -m "Sync with upstream commit abc123"
git push origin v1.1.0-upstream-sync
```

### 5. 文檔更新
- 每次適配後更新文檔
- 記錄變更原因和適配邏輯

---

## ✨ 總結

通過 Adapter Pattern:
- ✅ **隔離變更**: 上游更新只影響 Adapter 層
- ✅ **清晰邊界**: predict/ (CaiJianxiu) vs app/ (你)
- ✅ **快速同步**: `git merge` + 小量 Adapter 修改
- ✅ **降低風險**: Service 層不受上游變更影響

**記住**: predict/ 目錄永遠不要手動修改，只通過 git 同步！
