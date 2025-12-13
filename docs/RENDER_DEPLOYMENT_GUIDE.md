# Render 平台部署完整指南

**更新日期**: 2024-12-13  
**項目**: Multi-AOP FastAPI 微服務  
**版本**: v1.0.0

---

## 📋 目錄

1. [為什麼選擇 Render](#為什麼選擇-render)
2. [部署前準備](#部署前準備)
3. [方法一：使用 Blueprint（推薦）](#方法一使用-blueprint推薦)
4. [方法二：手動配置](#方法二手動配置)
5. [部署後驗證](#部署後驗證)
6. [處理休眠機制](#處理休眠機制)
7. [監控和日誌](#監控和日誌)
8. [常見問題](#常見問題)
9. [成本優化](#成本優化)

---

## 🎯 為什麼選擇 Render

### ✅ Render 的優勢

| 特性 | Render | Railway | 說明 |
|------|--------|---------|------|
| **永久免費** | ✅ 是 | ❌ 僅30天試用 | Render 提供真正的永久免費計劃 |
| **免費時長** | 750小時/月 | 30天試用 | 足夠 24/7 運行（31天） |
| **無需信用卡** | ✅ 是 | ✅ 是 | 註冊即可使用 |
| **部署限制** | ✅ 無限制 | ❌ 需要驗證 | GitHub 集成無需額外驗證 |
| **Docker 支持** | ✅ 優秀 | ✅ 優秀 | 原生 Docker 部署 |
| **穩定性** | ✅ 優秀 | ✅ 良好 | 企業級基礎設施 |
| **免費數據庫** | ✅ 1GB PostgreSQL | ❌ 無 | 未來擴展可用 |

### 📊 成本對比

| 使用場景 | Render | Railway |
|---------|--------|---------|
| **MVP/Demo** | **免費（永久）** | 免費（僅30天） |
| **生產環境** | $7/月起 | $5-20/月 |
| **高流量** | $15-30/月 | $20-50/月 |

---

## 🔧 部署前準備

### 1. 確認項目結構

確保以下文件存在：

```bash
Multi_AOP_FastAPI/
├── docker/
│   └── Dockerfile          # ✅ 已修改以包含模型文件
├── predict/
│   └── model/
│       └── best_model_Oct13.pth  # ✅ 模型文件（8.7MB）
├── app/                    # ✅ FastAPI 應用代碼
├── render.yaml             # ✅ Render 配置文件（新創建）
├── requirements.txt        # ✅ Python 依賴
└── .gitignore             # ✅ Git 忽略文件
```

### 2. 檢查模型文件

```bash
# 確認模型文件存在
ls -lh predict/model/best_model_Oct13.pth

# 預期輸出：
# -rw-r--r--  1 user  staff   8.7M Oct 13 12:00 best_model_Oct13.pth
```

### 3. 準備 Git 倉庫

```bash
# 如果還沒有初始化 Git
git init

# 添加所有文件
git add .

# 提交
git commit -m "feat: 準備 Render 部署配置"

# 推送到 GitHub（如果還沒有遠程倉庫）
git remote add origin https://github.com/your-username/Multi_AOP_FastAPI.git
git branch -M main
git push -u origin main
```

⚠️ **重要**：確保模型文件被提交到 Git（檢查 `.gitignore`）

---

## 🚀 方法一：使用 Blueprint（推薦）

### 優勢
- ✅ 一鍵部署
- ✅ 配置即代碼（Infrastructure as Code）
- ✅ 易於版本控制
- ✅ 可重複部署

### 步驟

#### 1. 註冊 Render 賬號

1. 訪問 [https://render.com](https://render.com)
2. 點擊 **"Get Started for Free"**
3. 使用 **GitHub 賬號**登錄（推薦）
4. ✅ **無需信用卡**

#### 2. 創建 Blueprint

1. 登錄後，點擊 **"New +"** → **"Blueprint"**
2. 選擇你的 GitHub 倉庫：`Multi_AOP_FastAPI`
3. Render 會自動檢測 `render.yaml` 文件
4. 點擊 **"Apply"**

#### 3. 等待部署

- 構建時間：約 **10-15 分鐘**（首次部署）
- Render 會自動：
  - 構建 Docker image
  - 安裝依賴（PyTorch, RDKit, xLSTM 等）
  - 部署應用
  - 配置健康檢查

#### 4. 獲取部署 URL

部署完成後，你會獲得一個 URL：

```
https://multi-aop-api.onrender.com
```

---

## 🛠️ 方法二：手動配置

如果不使用 Blueprint，可以手動配置：

### 步驟

#### 1. 創建 Web Service

1. 登錄 Render
2. 點擊 **"New +"** → **"Web Service"**
3. 連接 GitHub 倉庫

#### 2. 基本配置

| 設置項 | 值 |
|--------|-----|
| **Name** | `multi-aop-api` |
| **Region** | `Singapore`（或最近的區域） |
| **Branch** | `main` |
| **Root Directory** | `/`（項目根目錄） |
| **Runtime** | `Docker` |

#### 3. Docker 配置

| 設置項 | 值 |
|--------|-----|
| **Dockerfile Path** | `./docker/Dockerfile` |
| **Docker Context** | `.` |
| **Docker Command** | 留空（使用 Dockerfile 中的 CMD） |

#### 4. 環境變量配置

在 **"Environment"** 標籤中添加以下變量：

```bash
# API 配置
API_HOST=0.0.0.0
API_PORT=8000

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

#### 5. 高級設置

| 設置項 | 值 |
|--------|-----|
| **Instance Type** | `Free` |
| **Auto-Deploy** | `Yes`（啟用自動部署） |
| **Health Check Path** | `/health` |

#### 6. 創建服務

點擊 **"Create Web Service"**，等待部署完成。

---

## ✅ 部署後驗證

### 1. 健康檢查

```bash
# 替換為你的實際 URL
curl https://multi-aop-api.onrender.com/health
```

**預期響應**：

```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-12-13T10:30:00Z",
  "environment": "production",
  "message": ""
}
```

### 2. API 文檔

訪問 Swagger UI：

```
https://multi-aop-api.onrender.com/docs
```

### 3. 測試預測 API

#### 單個序列預測

```bash
curl -X POST "https://multi-aop-api.onrender.com/api/v1/predict/single" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MKLLVVVFCLVLAAP"
  }'
```

**預期響應**：

```json
{
  "sequence": "MKLLVVVFCLVLAAP",
  "prediction": 1,
  "probability": 0.85,
  "confidence": "high",
  "is_aop": true,
  "message": "Prediction completed successfully"
}
```

#### 批次預測

```bash
curl -X POST "https://multi-aop-api.onrender.com/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": [
      "MKLLVVVFCLVLAAP",
      "ACDEFGHIKLMNPQRSTVWY",
      "GPETLCGAELVDALQFVCGDRGFYFNKPTGYGSSSRRAPQT"
    ]
  }'
```

### 4. 查看日誌

在 Render Dashboard 中：
1. 選擇你的服務
2. 點擊 **"Logs"** 標籤
3. 查看實時日誌

---

## ⏰ 處理休眠機制

### 免費計劃的休眠問題

Render 免費計劃會在 **15 分鐘無活動**後自動休眠：
- ⚠️ 首次請求需要 **30秒-2分鐘**喚醒
- ⚠️ 影響用戶體驗

### 解決方案

#### 方案 1：接受休眠（推薦用於 MVP）

- ✅ 完全免費
- ✅ 無需額外配置
- ⚠️ 首次請求有延遲
- ✅ 適合 Demo/MVP

#### 方案 2：使用外部監控服務保持活躍

使用 **UptimeRobot**（免費）每 10 分鐘 ping 一次：

1. 註冊 [UptimeRobot](https://uptimerobot.com/)（免費）
2. 添加新監控：
   - **Monitor Type**: HTTP(s)
   - **URL**: `https://multi-aop-api.onrender.com/health`
   - **Monitoring Interval**: 10 分鐘
3. 保存

這樣可以保持服務活躍，避免休眠。

⚠️ **注意**：這會消耗更多的免費時長（750小時/月），但通常足夠。

#### 方案 3：升級到付費計劃

- **成本**: $7/月（Starter 計劃）
- ✅ 無休眠限制
- ✅ 更好的性能
- ✅ 更多資源
- ✅ 適合生產環境

---

## 📊 監控和日誌

### 1. Render Dashboard

在 Render Dashboard 中可以查看：
- ✅ 實時日誌
- ✅ 部署歷史
- ✅ 資源使用情況
- ✅ 健康檢查狀態

### 2. 日誌查看

```bash
# 在 Render Dashboard 中查看實時日誌
# 或使用 Render CLI（需要安裝）
render logs -f multi-aop-api
```

### 3. 監控指標

免費計劃提供基礎監控：
- CPU 使用率
- 內存使用率
- 請求數量
- 響應時間

---

## ❓ 常見問題

### Q1: 部署失敗，提示 "Model file not found"

**原因**：模型文件未包含在 Docker image 中。

**解決方案**：
1. 確認 `Dockerfile` 中已取消註釋 `COPY predict/model/ /app/predict/model/`
2. 確認模型文件已提交到 Git
3. 重新部署

### Q2: 構建時間過長（超過 15 分鐘）

**原因**：依賴包（PyTorch, RDKit）較大。

**解決方案**：
- ✅ 正常現象，首次構建需要 10-15 分鐘
- ✅ 後續部署會使用緩存，速度更快（2-5 分鐘）

### Q3: 服務啟動後立即崩潰

**原因**：可能是模型加載失敗或內存不足。

**解決方案**：
1. 查看日誌：`Logs` 標籤
2. 確認 `DEVICE=cpu`（免費計劃不支持 GPU）
3. 確認模型文件路徑正確

### Q4: API 請求超時

**原因**：服務休眠或推理時間過長。

**解決方案**：
1. 如果是首次請求，等待 30秒-2分鐘（喚醒時間）
2. 使用 UptimeRobot 保持服務活躍
3. 考慮升級到付費計劃

### Q5: 超過 750 小時/月限制

**原因**：服務持續運行超過 31 天。

**解決方案**：
1. 使用休眠機制（自動）
2. 升級到付費計劃（$7/月）
3. 優化服務使用時間

### Q6: CORS 錯誤

**原因**：CORS 配置不正確。

**解決方案**：
1. 確認環境變量 `CORS_ORIGINS=["*"]`（允許所有來源）
2. 或設置特定來源：`CORS_ORIGINS=["https://your-frontend.com"]`
3. 重新部署

---

## 💰 成本優化

### 免費計劃優化策略

#### 1. 使用休眠機制

- ✅ 15 分鐘無活動後自動休眠
- ✅ 節省免費時長
- ⚠️ 首次請求有延遲

#### 2. 優化請求頻率

- ✅ 批次處理多個序列（使用 `/api/v1/predict/batch`）
- ✅ 減少單個請求數量
- ✅ 提高效率

#### 3. 監控使用量

在 Render Dashboard 中查看：
- 已使用時長
- 剩餘時長
- 帶寬使用情況

### 何時升級到付費計劃？

考慮升級的情況：
- ✅ 需要 24/7 無休眠運行
- ✅ 超過 750 小時/月限制
- ✅ 需要更好的性能
- ✅ 生產環境部署

**付費計劃成本**：
- **Starter**: $7/月（512MB RAM，0.5 CPU）
- **Standard**: $25/月（2GB RAM，1 CPU）

---

## 🎯 總結

### ✅ Render 部署的優勢

1. **永久免費**：750 小時/月，足夠 MVP 使用
2. **簡單易用**：一鍵部署，自動化 CI/CD
3. **Docker 支持**：原生 Docker 部署，無需修改代碼
4. **穩定可靠**：企業級基礎設施
5. **無需信用卡**：註冊即可使用

### 📈 後續步驟

1. ✅ 部署到 Render 免費計劃
2. ✅ 測試 API 功能
3. ✅ 配置 UptimeRobot（可選）
4. ✅ 監控使用量
5. ✅ 根據需求考慮升級

### 🔗 相關資源

- [Render 官方文檔](https://render.com/docs)
- [FastAPI 部署指南](https://fastapi.tiangolo.com/deployment/)
- [Docker 最佳實踐](https://docs.docker.com/develop/dev-best-practices/)

---

**部署完成！** 🎉

如有問題，請查看 [常見問題](#常見問題) 或聯繫技術支持。

**最後更新**: 2024-12-13

