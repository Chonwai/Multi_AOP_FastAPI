# 部署方案分析：Vercel vs 其他平台

**分析日期**: 2024-12-19  
**項目**: Multi-AOP FastAPI 微服務  
**版本**: v1.0.0 MVP

---

## 📊 項目大小分析

### 代碼和模型大小

| 組件 | 大小 | 說明 |
|------|------|------|
| 模型文件 | 8.7MB × 2 | `best_model_Oct13.pth`, `best_combined_model.pth` |
| 應用代碼 | ~5MB | Python 源代碼 |
| 項目總大小 | ~30MB | 不包括數據目錄 |

### 依賴包大小估算

| 依賴包 | 估算大小 | 說明 |
|--------|---------|------|
| PyTorch (CPU) | ~500-800MB | 核心深度學習框架 |
| torch-geometric | ~50-100MB | 圖神經網絡擴展 |
| RDKit | ~200-300MB | 化學信息學庫（conda安裝） |
| xLSTM | ~10-20MB | xLSTM 實現 |
| FastAPI + uvicorn | ~10-20MB | Web 框架和 ASGI 服務器 |
| 其他依賴 | ~50-100MB | pandas, numpy, pydantic 等 |
| **總計** | **~800MB-1.2GB** | **未壓縮大小** |

**關鍵發現**: 依賴包總大小遠超過 Vercel 的限制！

---

## 🚫 Vercel 部署可行性分析

### Vercel Serverless Functions 限制

| 限制項 | Hobby 計劃 | Pro 計劃 | 我們的項目 |
|--------|-----------|---------|-----------|
| **未壓縮大小** | 250MB | 250MB | ❌ **~800MB-1.2GB** |
| **壓縮後大小** | 50MB | 50MB | ❌ **~200-300MB** |
| **執行時間** | 10秒 | 60秒 | ⚠️ **可能超時**（推理需1-3秒） |
| **內存** | 1024MB | 1024MB | ⚠️ **可能不足**（PyTorch推理） |
| **冷啟動** | 無限制 | 無限制 | ⚠️ **可能很慢**（大型依賴） |

### ❌ Vercel 不適合的原因

1. **大小限制超標**
   - 依賴包總大小 ~800MB-1.2GB，遠超 250MB 限制
   - 即使壓縮後也可能超過 50MB 限制

2. **執行時間限制**
   - ML 推理需要 1-3 秒
   - 加上冷啟動時間，可能接近或超過 10 秒限制（Hobby）
   - Pro 計劃的 60 秒可能足夠，但成本較高

3. **冷啟動問題**
   - 大型依賴包（PyTorch, RDKit）冷啟動時間長
   - 用戶體驗差（首次請求可能需 10-30 秒）

4. **內存限制**
   - PyTorch 模型推理可能需要 1-2GB 內存
   - Vercel 的 1024MB 可能不夠

5. **架構不匹配**
   - Vercel 設計用於輕量級 Serverless Functions
   - ML 推理服務需要長時間運行的容器

---

## ✅ 推薦的替代部署方案

### 1. Railway（⭐ 強烈推薦）

**優勢**:
- ✅ 支持 Docker 部署（我們已有 Dockerfile）
- ✅ 無大小限制
- ✅ 自動部署（GitHub 集成）
- ✅ 免費計劃：$5 信用額度/月
- ✅ 簡單易用

**配置**:
```yaml
# railway.json
{
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "docker/Dockerfile"
  },
  "deploy": {
    "startCommand": "conda run -n app uvicorn app.main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

**成本**: 免費計劃足夠 MVP，付費計劃 $5-20/月

---

### 2. Render

**優勢**:
- ✅ 支持 Docker 部署
- ✅ 免費計劃可用（有限制）
- ✅ 自動 HTTPS
- ✅ 簡單配置

**配置**:
```yaml
# render.yaml
services:
  - type: web
    name: multi-aop-api
    dockerfilePath: docker/Dockerfile
    env: python
    envVars:
      - key: MODEL_PATH
        value: predict/model/best_model_Oct13.pth
```

**成本**: 免費計劃（有限制），付費計劃 $7-25/月

---

### 3. Fly.io

**優勢**:
- ✅ 全球邊緣部署
- ✅ 支持 Docker
- ✅ 免費計劃：3 個共享 CPU 實例
- ✅ 優秀的開發者體驗

**配置**:
```toml
# fly.toml
app = "multi-aop-api"
primary_region = "hkg"

[build]
  dockerfile = "docker/Dockerfile"

[[services]]
  internal_port = 8000
  protocol = "tcp"
```

**成本**: 免費計劃可用，付費計劃按使用量計費

---

### 4. Google Cloud Run

**優勢**:
- ✅ 完全託管
- ✅ 按請求計費（無請求時不計費）
- ✅ 自動擴展
- ✅ 支持 Docker

**配置**:
```bash
# 使用 gcloud CLI
gcloud run deploy multi-aop-api \
  --source . \
  --platform managed \
  --region asia-east1 \
  --memory 2Gi \
  --timeout 300
```

**成本**: 按使用量計費，免費額度充足

---

### 5. AWS Lambda + ECS/Fargate

**優勢**:
- ✅ 企業級可靠性
- ✅ 高度可擴展
- ⚠️ 配置複雜

**成本**: 按使用量計費，可能較高

---

## 📋 部署方案對比表

| 平台 | Docker支持 | 免費計劃 | 易用性 | 成本 | 推薦度 |
|------|-----------|---------|--------|------|--------|
| **Railway** | ✅ | ✅ ($5/月) | ⭐⭐⭐⭐⭐ | 低 | ⭐⭐⭐⭐⭐ |
| **Render** | ✅ | ✅ (有限制) | ⭐⭐⭐⭐ | 低-中 | ⭐⭐⭐⭐ |
| **Fly.io** | ✅ | ✅ (3實例) | ⭐⭐⭐⭐ | 低 | ⭐⭐⭐⭐ |
| **Google Cloud Run** | ✅ | ✅ (免費額度) | ⭐⭐⭐ | 低-中 | ⭐⭐⭐⭐ |
| **AWS Lambda/ECS** | ✅ | ❌ | ⭐⭐ | 中-高 | ⭐⭐⭐ |
| **Vercel** | ❌ | ✅ | ⭐⭐⭐⭐⭐ | 低 | ❌ 不適合 |

---

## 🎯 最終建議

### 對於 MVP/Demo 版本

**推薦方案**: **Railway** 或 **Render**

**理由**:
1. ✅ 支持 Docker（我們已有完整配置）
2. ✅ 免費計劃足夠 MVP 使用
3. ✅ 配置簡單，部署快速
4. ✅ 無大小限制
5. ✅ 自動 HTTPS 和域名

### 部署步驟（Railway 示例）

1. **準備部署**:
   ```bash
   # 確保 Dockerfile 已準備好
   cd docker
   docker build -f Dockerfile -t multi-aop-api ..
   ```

2. **Railway 部署**:
   - 註冊 Railway 賬號
   - 連接 GitHub 倉庫
   - 選擇 Dockerfile 路徑：`docker/Dockerfile`
   - 設置環境變量（從 `.env.example`）
   - 部署！

3. **驗證部署**:
   ```bash
   curl https://your-app.railway.app/health
   ```

---

## 🔧 如果必須使用 Vercel（不推薦）

如果堅持使用 Vercel，需要重大架構調整：

### 方案 A: 分離架構

1. **前端 + API Gateway**: 部署在 Vercel
2. **ML 推理服務**: 部署在其他平台（Railway/Render）
3. **API 調用**: Vercel Functions 調用外部 ML 服務

**缺點**: 
- 增加複雜度
- 增加延遲（額外的網絡調用）
- 需要維護兩個服務

### 方案 B: 模型優化

1. **量化模型**: 減少模型大小
2. **移除 RDKit**: 使用預處理服務（不推薦，影響功能）
3. **使用 TensorFlow Lite**: 更小的運行時（需要重寫代碼）

**缺點**:
- 需要大量代碼修改
- 可能影響預測準確性
- 開發時間長

---

## 📝 總結

### ✅ 結論

**Vercel 不適合部署此項目**，原因：
1. ❌ 依賴包大小（~800MB-1.2GB）遠超限制（250MB）
2. ❌ 執行時間限制可能不足
3. ❌ 冷啟動時間長
4. ❌ 內存可能不足

### 🎯 推薦行動

1. **短期（MVP）**: 使用 **Railway** 或 **Render**
   - 快速部署
   - 免費計劃可用
   - 支持 Docker

2. **長期（生產）**: 考慮 **Google Cloud Run** 或 **AWS ECS**
   - 更好的可擴展性
   - 企業級可靠性
   - 按使用量計費

---

**分析完成日期**: 2024-12-19  
**建議**: 使用 Railway 或 Render 進行 MVP 部署

