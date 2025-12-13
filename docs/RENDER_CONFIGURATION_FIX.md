# Render 配置修正指南

**日期**: 2024-12-13  
**問題**: Render 部署配置錯誤  
**狀態**: 🔴 需要立即修正

---

## 🚨 發現的關鍵問題

從你的截圖分析，發現以下**嚴重配置錯誤**：

### 問題 1：Dockerfile Path 未指定 🔴

**問題描述**:
- 你的 Dockerfile 位於 `docker/Dockerfile`（非根目錄）
- Render 無法自動檢測到這個路徑
- 會導致構建失敗

**解決方案**:
```
在 Render Dashboard 中設置：
Dockerfile Path: docker/Dockerfile
```

---

### 問題 2：Environment 未設置為 Docker 🔴

**問題描述**:
- 必須明確告訴 Render 這是一個 Docker 部署
- 如果未設置，Render 會嘗試自動檢測語言，可能導致錯誤

**解決方案**:
```
在 Render Dashboard 中設置：
Environment: Docker
```

---

### 問題 3：端口配置錯誤 🔴

**問題描述**:
- Render 默認使用 `PORT=10000`
- 但你的 Dockerfile CMD 使用固定的 `8000` 端口
- 會導致 Render 無法連接到你的應用

**當前 Dockerfile (錯誤)**:
```dockerfile
CMD ["conda", "run", "-n", "app", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**修正後的 Dockerfile (正確)**:
```dockerfile
CMD ["conda", "run", "-n", "app", "sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

---

### 問題 4：Docker Context 未確認 🟡

**問題描述**:
- Docker Context 決定 COPY 指令的相對路徑
- 必須是項目根目錄（因為 Dockerfile 中 COPY 使用根目錄路徑）

**解決方案**:
```
在 Render Dashboard 中設置：
Docker Context: .
```

---

### 問題 5：Health Check 未設置 🟡

**問題描述**:
- Render 需要知道如何檢查服務健康狀態
- 未設置可能導致服務被標記為不健康

**解決方案**:
```
在 Render Dashboard 中設置：
Health Check Path: /health
```

---

## ✅ 完整修正步驟

### 步驟 1：修改 Dockerfile

首先修改 `docker/Dockerfile` 以支持 Render 的 PORT 環境變量：

```dockerfile
# 找到最後的 CMD 行（第 138 行）
# 從：
CMD ["conda", "run", "-n", "app", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# 改為：
CMD ["conda", "run", "-n", "app", "sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

**解釋**:
- `${PORT:-8000}`: 使用 Render 的 PORT 環境變量，如果未設置則默認 8000
- `sh -c`: 允許 shell 變量替換

---

### 步驟 2：在 Render Dashboard 中配置

#### 2.1 基本設置

| 設置項 | 值 | 說明 |
|--------|-----|------|
| **Name** | `multi-aop-api` | 服務名稱 |
| **Region** | `Singapore` | 選擇最近的區域 |
| **Branch** | `main` | ✅ 已正確 |
| **Root Directory** | (留空) | ✅ 使用項目根目錄 |

#### 2.2 構建設置（🔴 關鍵）

| 設置項 | 值 | 說明 |
|--------|-----|------|
| **Environment** | `Docker` | 🔴 必須設置 |
| **Dockerfile Path** | `docker/Dockerfile` | 🔴 必須設置 |
| **Docker Context** | `.` | 項目根目錄 |
| **Docker Command** | (留空) | 使用 Dockerfile 中的 CMD |

#### 2.3 環境變量設置

在 **Environment Variables** 部分添加以下變量：

```bash
# API 配置
API_HOST=0.0.0.0
API_PORT=8000  # 注意：實際運行時會被 $PORT 覆蓋

# CORS 配置
CORS_ORIGINS=["*"]

# 模型配置
MODEL_PATH=predict/model/best_model_Oct13.pth
DEVICE=cpu

# 序列處理配置
SEQ_LENGTH=50
BATCH_SIZE=16
MAX_BATCH_SIZE=100

# 日誌配置
LOG_LEVEL=INFO

# 環境
ENVIRONMENT=production
```

⚠️ **重要**：不需要手動設置 `PORT` 環境變量，Render 會自動設置為 `10000`

#### 2.4 高級設置

| 設置項 | 值 | 說明 |
|--------|-----|------|
| **Instance Type** | `Free` | ✅ 免費計劃 |
| **Auto-Deploy** | `Yes` | ✅ 啟用自動部署 |
| **Health Check Path** | `/health` | 健康檢查端點 |

---

### 步驟 3：提交更改到 Git

```bash
# 修改 Dockerfile 後
cd /path/to/Multi_AOP_FastAPI

# 查看更改
git diff docker/Dockerfile

# 添加更改
git add docker/Dockerfile

# 提交
git commit -m "fix: 修改 Dockerfile 以支持 Render PORT 環境變量"

# 推送到 GitHub
git push origin main
```

---

### 步驟 4：在 Render 中重新部署

1. 在 Render Dashboard 中找到你的服務
2. 點擊 **"Manual Deploy"** → **"Deploy latest commit"**
3. 或者等待自動部署（如果啟用了 Auto-Deploy）

---

## 🔍 配置檢查清單

在部署前，請確認以下所有項目：

### 基本配置

- [ ] Repository 正確：`https://github.com/chonwai-y/Multi_AOP_FastAPI`
- [ ] Branch 正確：`main`
- [ ] Root Directory 為空（使用根目錄）

### Docker 配置（🔴 關鍵）

- [ ] **Environment 設置為 `Docker`**
- [ ] **Dockerfile Path 設置為 `docker/Dockerfile`**
- [ ] **Docker Context 設置為 `.`**
- [ ] Docker Command 留空（使用 Dockerfile CMD）

### Dockerfile 修改

- [ ] **CMD 使用 `${PORT:-8000}` 而不是固定的 `8000`**
- [ ] Dockerfile 包含模型文件：`COPY predict/model/ /app/predict/model/`
- [ ] Dockerfile 使用非 root 用戶：`USER appuser`

### 環境變量

- [ ] API_HOST=0.0.0.0
- [ ] CORS_ORIGINS=["*"]
- [ ] MODEL_PATH=predict/model/best_model_Oct13.pth
- [ ] DEVICE=cpu
- [ ] SEQ_LENGTH=50
- [ ] BATCH_SIZE=16
- [ ] MAX_BATCH_SIZE=100
- [ ] LOG_LEVEL=INFO
- [ ] ENVIRONMENT=production

### 高級設置

- [ ] Instance Type: Free
- [ ] Auto-Deploy: Yes
- [ ] Health Check Path: `/health`

---

## ⚠️ 常見錯誤與解決方案

### 錯誤 1：構建失敗 "Dockerfile not found"

**原因**: Dockerfile Path 未設置或設置錯誤

**解決方案**:
```
確保 Dockerfile Path 設置為: docker/Dockerfile
注意：不是 /docker/Dockerfile，不要加前導斜杠
```

---

### 錯誤 2：服務啟動但 Render 顯示 "Service Unavailable"

**原因**: 端口配置錯誤，應用監聽 8000 但 Render 期望 10000

**解決方案**:
```dockerfile
# 修改 Dockerfile CMD 以使用 $PORT
CMD ["conda", "run", "-n", "app", "sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

---

### 錯誤 3：構建超時或失敗 "Out of memory"

**原因**: 依賴包過大（PyTorch, RDKit 等）

**解決方案**:
- 正常現象，首次構建需要 10-15 分鐘
- 如果持續失敗，考慮使用更小的基礎鏡像
- 或升級到付費計劃（更多內存）

---

### 錯誤 4：模型文件未找到

**原因**: 
1. 模型文件未包含在 Docker image 中
2. 模型文件未提交到 Git

**解決方案**:
```bash
# 檢查模型文件是否在 Git 中
git ls-files predict/model/

# 如果沒有，強制添加
git add -f predict/model/best_model_Oct13.pth
git commit -m "fix: 添加模型文件"
git push

# 確認 Dockerfile 包含 COPY 指令
# COPY predict/model/ /app/predict/model/
```

---

## 🧪 本地測試

在推送到 Render 之前，建議先在本地測試：

### 測試 1：構建 Docker Image

```bash
cd /path/to/Multi_AOP_FastAPI

# 構建
docker build -f docker/Dockerfile -t multi-aop-test .

# 檢查 image 大小
docker images multi-aop-test
# 預期：約 1.2GB
```

### 測試 2：測試 PORT 環境變量

```bash
# 測試默認端口（8000）
docker run -p 8000:8000 multi-aop-test

# 測試 Render 的端口（10000）
docker run -p 10000:10000 -e PORT=10000 multi-aop-test

# 測試健康檢查
curl http://localhost:10000/health
```

### 測試 3：測試 API 功能

```bash
# 單序列預測
curl -X POST "http://localhost:10000/api/v1/predict/single" \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKLLVVVFCLVLAAP"}'

# 預期響應：
# {
#   "sequence": "MKLLVVVFCLVLAAP",
#   "prediction": 1,
#   "probability": 0.85,
#   "confidence": "high",
#   "is_aop": true
# }
```

---

## 📊 修正前後對比

### 修正前（❌ 錯誤配置）

```yaml
Environment: (未設置或自動檢測)
Dockerfile Path: (未設置)
Docker Context: (未設置)
Docker Command: (未設置)

# Dockerfile CMD
CMD ["conda", "run", "-n", "app", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**結果**: 🔴 部署失敗或無法連接

---

### 修正後（✅ 正確配置）

```yaml
Environment: Docker  ✅
Dockerfile Path: docker/Dockerfile  ✅
Docker Context: .  ✅
Docker Command: (留空，使用 Dockerfile CMD)  ✅
Health Check Path: /health  ✅

# Dockerfile CMD
CMD ["conda", "run", "-n", "app", "sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

**結果**: ✅ 部署成功

---

## 🚀 部署後驗證

部署完成後，執行以下檢查：

### 1. 檢查服務狀態

```bash
# 在 Render Dashboard 中
# 狀態應該顯示: ✅ Live (綠色)
```

### 2. 健康檢查

```bash
curl https://your-app.onrender.com/health

# 預期響應：
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-12-13T10:30:00Z",
  "environment": "production"
}
```

### 3. API 測試

```bash
curl -X POST "https://your-app.onrender.com/api/v1/predict/single" \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKLLVVVFCLVLAAP"}'
```

### 4. 查看日誌

在 Render Dashboard 中查看日誌，確認：
- ✅ 模型成功加載
- ✅ 應用監聽正確端口（應該顯示 `PORT=10000`）
- ✅ 無錯誤信息

---

## 💡 專業建議

### 1. 使用 render.yaml（推薦）

為了避免手動配置錯誤，建議使用 `render.yaml`：

```yaml
services:
  - type: web
    name: multi-aop-api
    runtime: docker
    dockerfilePath: ./docker/Dockerfile
    dockerContext: .
    plan: free
    healthCheckPath: /health
    envVars:
      - key: API_HOST
        value: 0.0.0.0
      - key: MODEL_PATH
        value: predict/model/best_model_Oct13.pth
      # ... 其他環境變量
```

### 2. 監控和告警

設置 UptimeRobot 監控：
- URL: `https://your-app.onrender.com/health`
- 間隔: 10 分鐘
- 防止服務休眠

### 3. 成本優化

免費計劃限制：
- ⏰ 750 小時/月
- 📡 100GB 出站帶寬
- ⚠️ 15 分鐘無活動後休眠

如果需要 24/7 運行且無休眠，考慮升級到付費計劃（$7/月）

---

## 🔗 相關資源

- [Render Docker 官方文檔](https://render.com/docs/docker)
- [Render 環境變量文檔](https://docs.render.com/environment-variables)
- [項目部署完整指南](./RENDER_DEPLOYMENT_GUIDE.md)
- [項目架構分析](./PROJECT_ANALYSIS.md)

---

## ✅ 總結

### 必須修改的地方：

1. 🔴 **Dockerfile CMD**：改為使用 `${PORT:-8000}`
2. 🔴 **Render Environment**：設置為 `Docker`
3. 🔴 **Dockerfile Path**：設置為 `docker/Dockerfile`
4. 🟡 **Docker Context**：設置為 `.`
5. 🟡 **Health Check Path**：設置為 `/health`

### 修改優先級：

1. **立即修改**（部署會失敗）：
   - Dockerfile CMD 端口配置
   - Render Environment 設置
   - Dockerfile Path 設置

2. **強烈建議修改**（可能導致問題）：
   - Docker Context 設置
   - Health Check Path 設置

3. **可選修改**（優化）：
   - 使用 render.yaml 自動化配置
   - 設置 UptimeRobot 監控

---

**修正完成後，你的應用應該能夠成功部署到 Render！** 🎉

如有問題，請查看日誌或參考常見錯誤部分。

**最後更新**: 2024-12-13

