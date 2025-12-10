# 部署平台最終推薦

**更新日期**: 2024-12-19  
**項目**: Multi-AOP FastAPI 微服務  
**版本**: v1.0.0 MVP

---

## ⚠️ 重要更新：Railway 免費計劃變更

### Railway 實際情況

根據最新信息確認：

1. **❌ 不是永久免費**
   - Railway **僅提供30天免費試用**
   - 試用期結束後**必須付費**
   - 2023年8月已取消永久免費計劃

2. **❌ 部署限制**
   - 免費用戶可能**無法部署代碼**
   - 需要GitHub賬號驗證通過
   - 如果驗證失敗，無法使用部署功能

3. **💰 30天後成本**
   - 必須升級到付費計劃
   - 最低約 $5-20/月

---

## ✅ 最終推薦：Render

### 為什麼選擇 Render？

| 特性 | Render | Railway |
|------|--------|---------|
| **永久免費** | ✅ 是 | ❌ 僅30天試用 |
| **無需信用卡** | ✅ 是 | ✅ 是 |
| **部署限制** | ✅ 無限制 | ❌ 需要驗證 |
| **免費時長** | ✅ 750小時/月 | ⚠️ 30天試用 |
| **商業使用** | ✅ 允許 | ✅ 允許 |
| **穩定性** | ✅ 優秀 | ✅ 良好 |

---

## 🚀 Render 免費計劃詳情

### 免費額度（永久有效）

- ⏰ **750小時/月**（約31天，足夠24/7運行）
- 💾 **1GB PostgreSQL** 數據庫
- 📡 **100GB** 出站帶寬（每月重置）
- 🔄 自動部署（GitHub集成）
- 📊 基礎監控和日誌
- ✅ **無需信用卡**
- ✅ **永久免費**（never expires）

### 限制

- ⚠️ **休眠機制**：15分鐘無活動後自動休眠
- ⚠️ **喚醒延遲**：休眠後首次請求需等待30秒-2分鐘
- ⚠️ **月度限制**：超過750小時後服務暫停（下月自動重置）
- ⚠️ **帶寬限制**：超過100GB後服務暫停（下月自動重置）

### 成本

- **免費計劃**：**完全免費**，永久有效
- **付費計劃**：$7/月起（無休眠，更好的性能）

---

## 📋 部署步驟（Render）

### 1. 註冊 Render 賬號

1. 訪問 https://render.com
2. 點擊 "Get Started for Free"
3. 使用 GitHub 賬號登錄（推薦）
4. **無需信用卡**

### 2. 創建新的 Web Service

1. 點擊 "New +" → "Web Service"
2. 連接 GitHub 倉庫
3. 選擇你的倉庫：`Multi_AOP_FastAPI`

### 3. 配置服務

**基本設置**:
- **Name**: `multi-aop-api`
- **Region**: 選擇最近的區域（如 `Singapore`）
- **Branch**: `main` 或 `master`
- **Root Directory**: `/`（項目根目錄）

**構建設置**:
- **Build Command**: 
  ```bash
  docker build -f docker/Dockerfile -t app .
  ```
- **Start Command**: 
  ```bash
  docker run -p $PORT:8000 app
  ```
  
  或者更簡單的方式：
  - **Environment**: `Docker`
  - Render 會自動檢測 Dockerfile

**環境變量**（從 `env.example`）:
```
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["*"]
MODEL_PATH=predict/model/best_model_Oct13.pth
DEVICE=cpu
SEQ_LENGTH=50
BATCH_SIZE=16
MAX_BATCH_SIZE=100
LOG_LEVEL=INFO
ENVIRONMENT=production
```

**資源設置**:
- **Instance Type**: `Free`（免費計劃）
- **Auto-Deploy**: `Yes`（自動部署）

### 4. 部署

1. 點擊 "Create Web Service"
2. Render 會自動構建和部署
3. 等待構建完成（約5-10分鐘）
4. 部署完成後會獲得一個 URL：`https://your-app.onrender.com`

### 5. 驗證部署

```bash
# 健康檢查
curl https://your-app.onrender.com/health

# API 測試
curl -X POST "https://your-app.onrender.com/api/v1/predict/single" \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKLLVVVFCLVLAAP"}'
```

---

## 🔧 處理休眠機制

### 免費計劃的休眠問題

Render 免費計劃會在15分鐘無活動後休眠，首次請求需要30秒-2分鐘喚醒。

### 解決方案

#### 方案 1：接受休眠（推薦用於 MVP）

- ✅ 免費且簡單
- ⚠️ 首次請求有延遲
- ✅ 適合 Demo/MVP

#### 方案 2：使用外部服務保持活躍

使用免費的監控服務（如 UptimeRobot）每10分鐘 ping 一次：

1. 註冊 UptimeRobot（免費）
2. 添加監控 URL：`https://your-app.onrender.com/health`
3. 設置監控間隔：每10分鐘
4. 這樣可以保持服務活躍

#### 方案 3：升級到付費計劃

- 付費計劃（$7/月）無休眠限制
- 更好的性能
- 適合生產環境

---

## 📊 成本對比

| 方案 | 成本 | 說明 |
|------|------|------|
| **Render 免費計劃** | **$0/月** | 永久免費，750小時/月 |
| **Render 付費計劃** | **$7/月** | 無休眠，更好的性能 |
| **Railway 試用** | **$0（30天）** | 僅30天試用 |
| **Railway 付費** | **$5-20/月** | 30天後必須付費 |

---

## ✅ 最終建議

### 對於 MVP/Demo 版本

**強烈推薦：Render 免費計劃**

**理由**:
1. ✅ **永久免費**（never expires）
2. ✅ **無需信用卡**
3. ✅ **無部署限制**
4. ✅ 750小時/月足夠24/7運行
5. ✅ 優秀的穩定性
6. ✅ 免費PostgreSQL（未來擴展）

**注意事項**:
- ⚠️ 15分鐘休眠機制（可接受，或使用UptimeRobot保持活躍）
- ⚠️ 喚醒延遲（30秒-2分鐘）

### 對於生產環境

**推薦：Render 付費計劃（$7/月）**

**理由**:
1. ✅ 無休眠限制
2. ✅ 更好的性能
3. ✅ 成本可預測
4. ✅ 優秀的穩定性

---

## 🎯 總結

### ❌ Railway：不推薦

- ❌ 僅30天試用
- ❌ 部署限制（需要GitHub驗證）
- ❌ 30天後必須付費

### ✅ Render：強烈推薦

- ✅ **永久免費**
- ✅ **無需信用卡**
- ✅ **無部署限制**
- ✅ 750小時/月足夠使用
- ✅ 優秀的穩定性

---

**最終建議**: **使用 Render 免費計劃進行 MVP 部署**

**完成日期**: 2024-12-19

