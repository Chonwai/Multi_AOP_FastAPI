# Render 部署檢查清單 ✅

**項目**: Multi-AOP FastAPI  
**目標平台**: Render.com  
**部署類型**: Docker

---

## 📋 部署前檢查

### 1. 文件準備 ✅

- [x] `docker/Dockerfile` 已修改以包含模型文件
- [x] `render.yaml` 配置文件已創建
- [x] `.gitignore` 已更新以包含模型文件
- [x] `predict/model/best_model_Oct13.pth` 模型文件存在（8.7MB）
- [x] `requirements.txt` 依賴清單完整
- [x] `docs/RENDER_DEPLOYMENT_GUIDE.md` 部署指南已創建
- [x] `docs/PROJECT_ANALYSIS.md` 項目分析已完成

### 2. Git 倉庫準備

```bash
# 檢查 Git 狀態
git status

# 添加所有更改
git add .

# 提交更改
git commit -m "feat: 添加 Render 部署配置和文檔"

# 推送到 GitHub
git push origin main
```

### 3. 模型文件檢查

```bash
# 確認模型文件存在
ls -lh predict/model/best_model_Oct13.pth

# 確認模型文件會被 Git 追蹤
git check-ignore predict/model/best_model_Oct13.pth
# 如果輸出為空，說明文件會被追蹤 ✅
```

### 4. Docker 構建測試（可選但推薦）

```bash
# 在本地測試 Docker 構建
cd docker
docker build -f Dockerfile -t multi-aop-api:test ..

# 測試運行
docker run -p 8000:8000 -e DEVICE=cpu multi-aop-api:test

# 測試 API
curl http://localhost:8000/health
```

---

## 🚀 Render 部署步驟

### 方法一：使用 Blueprint（推薦）⭐

1. **註冊 Render**
   - 訪問 https://render.com
   - 使用 GitHub 賬號登錄
   - ✅ 無需信用卡

2. **創建 Blueprint**
   - 點擊 "New +" → "Blueprint"
   - 選擇倉庫：`Multi_AOP_FastAPI`
   - Render 自動檢測 `render.yaml`
   - 點擊 "Apply"

3. **等待部署**
   - 構建時間：10-15 分鐘
   - 查看構建日誌
   - 等待狀態變為 "Live"

4. **獲取 URL**
   - 部署完成後獲得 URL
   - 格式：`https://multi-aop-api.onrender.com`

### 方法二：手動配置

詳見 [`docs/RENDER_DEPLOYMENT_GUIDE.md`](./docs/RENDER_DEPLOYMENT_GUIDE.md)

---

## ✅ 部署後驗證

### 1. 健康檢查

```bash
# 替換為你的實際 URL
export API_URL="https://multi-aop-api.onrender.com"

# 健康檢查
curl $API_URL/health

# 預期響應：
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "timestamp": "2024-12-13T10:30:00Z",
#   "environment": "production"
# }
```

### 2. API 文檔測試

```bash
# 訪問 Swagger UI
open $API_URL/docs

# 或
open $API_URL/redoc
```

### 3. 功能測試

#### 測試單序列預測

```bash
curl -X POST "$API_URL/api/v1/predict/single" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MKLLVVVFCLVLAAP"
  }'

# 預期響應：
# {
#   "sequence": "MKLLVVVFCLVLAAP",
#   "prediction": 1,
#   "probability": 0.85,
#   "confidence": "high",
#   "is_aop": true,
#   "message": "Prediction completed successfully"
# }
```

#### 測試批次預測

```bash
curl -X POST "$API_URL/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": [
      "MKLLVVVFCLVLAAP",
      "ACDEFGHIKLMNPQRSTVWY",
      "GPETLCGAELVDALQFVCGDRGFYFNKPTGYGSSSRRAPQT"
    ]
  }'

# 預期響應：
# {
#   "total": 3,
#   "results": [...],
#   "processing_time_seconds": 0.5
# }
```

#### 測試模型信息

```bash
curl $API_URL/api/v1/model/info

# 預期響應：
# {
#   "model_version": "1.0.0",
#   "model_path": "predict/model/best_model_Oct13.pth",
#   "device": "cpu",
#   "seq_length": 50,
#   "loaded_at": "2024-12-13T10:30:00Z",
#   "is_loaded": true
# }
```

### 4. 性能測試

```bash
# 測試響應時間
time curl -X POST "$API_URL/api/v1/predict/single" \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKLLVVVFCLVLAAP"}'

# 預期：
# - 首次請求（喚醒）：30秒-2分鐘
# - 後續請求：100-200ms
```

---

## 📊 監控和維護

### 1. 查看日誌

在 Render Dashboard 中：
- 選擇服務 "multi-aop-api"
- 點擊 "Logs" 標籤
- 查看實時日誌

### 2. 監控資源使用

在 Render Dashboard 中查看：
- CPU 使用率
- 內存使用率
- 請求數量
- 已使用時長（750小時/月）

### 3. 設置 UptimeRobot（可選）

防止服務休眠：

1. 註冊 https://uptimerobot.com（免費）
2. 添加監控：
   - URL: `$API_URL/health`
   - 間隔: 10 分鐘
3. 保存

---

## 🔧 常見問題排查

### 問題 1: 部署失敗，提示 "Model file not found"

**解決方案**:
```bash
# 檢查模型文件是否在 Git 中
git ls-files predict/model/

# 如果沒有，強制添加
git add -f predict/model/best_model_Oct13.pth
git commit -m "fix: 添加模型文件"
git push
```

### 問題 2: 構建超時

**解決方案**:
- 正常現象，首次構建需要 10-15 分鐘
- 等待構建完成
- 查看構建日誌確認進度

### 問題 3: 服務啟動失敗

**解決方案**:
```bash
# 查看 Render 日誌
# 常見原因：
# 1. 環境變量配置錯誤
# 2. 模型文件路徑錯誤
# 3. 端口配置錯誤

# 檢查環境變量
# 確保 MODEL_PATH=predict/model/best_model_Oct13.pth
# 確保 DEVICE=cpu
```

### 問題 4: API 請求超時

**解決方案**:
- 首次請求需要 30秒-2分鐘（喚醒時間）
- 使用 UptimeRobot 保持服務活躍
- 或升級到付費計劃（$7/月）

---

## 💰 成本管理

### 免費計劃限制

- ⏰ **750 小時/月**（約 31 天）
- 📡 **100GB 出站帶寬/月**
- ⚠️ **15 分鐘無活動後休眠**

### 監控使用量

在 Render Dashboard 中查看：
- 已使用時長
- 剩餘時長
- 帶寬使用情況

### 升級時機

考慮升級到付費計劃（$7/月）的情況：
- ✅ 需要 24/7 無休眠運行
- ✅ 超過 750 小時/月
- ✅ 需要更好的性能
- ✅ 生產環境部署

---

## 📚 相關文檔

- [Render 部署完整指南](./docs/RENDER_DEPLOYMENT_GUIDE.md)
- [項目深度分析](./docs/PROJECT_ANALYSIS.md)
- [Railway vs Render 對比](./docs/development/v1.0.0-fastapi-microservice/RAILWAY_VS_RENDER.md)
- [前端開發指南](./docs/development/v1.0.0-fastapi-microservice/FRONTEND_GUIDE.md)

---

## ✅ 部署完成檢查

部署成功的標誌：

- [x] 服務狀態為 "Live"
- [x] 健康檢查返回 `"status": "healthy"`
- [x] API 文檔可訪問（`/docs`）
- [x] 單序列預測正常工作
- [x] 批次預測正常工作
- [x] 模型信息可獲取
- [x] 響應時間正常（100-200ms）

---

## 🎉 下一步

部署完成後：

1. ✅ 分享 API URL 給團隊
2. ✅ 測試所有功能
3. ✅ 收集用戶反饋
4. ✅ 監控使用情況
5. ✅ 根據需求優化

---

**祝部署順利！** 🚀

如有問題，請查看 [Render 部署指南](./docs/RENDER_DEPLOYMENT_GUIDE.md) 或聯繫技術支持。

**最後更新**: 2024-12-13

