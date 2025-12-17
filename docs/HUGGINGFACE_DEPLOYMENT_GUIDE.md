# HuggingFace Spaces 部署完整指南

## 📋 目錄
- [概述](#概述)
- [前置準備](#前置準備)
- [部署步驟](#部署步驟)
- [問題排查](#問題排查)
- [技術細節](#技術細節)
- [本地測試](#本地測試)

---

## 🎯 概述

本指南詳細說明如何將 **Multi-AOP FastAPI** 微服務部署到 HuggingFace Spaces 平台。

### 部署架構

```
GitHub Repository (production branch)
    ↓ (GitHub Actions)
HuggingFace Space (Docker SDK)
    ↓ (Auto-build & Deploy)
Public API Endpoint
```

### 技術棧
- **平台**: HuggingFace Spaces
- **SDK**: Docker
- **CI/CD**: GitHub Actions
- **大文件管理**: Git LFS (Large File Storage)

---

## ✅ 前置準備

### 1. HuggingFace 帳號設置

1. 註冊 [HuggingFace](https://huggingface.co/) 帳號
2. 創建新的 Space:
   - 名稱: `Multi_AOP_FastAPI`
   - License: 根據項目需求選擇
   - **SDK: Docker** ⚠️ 這是關鍵設置！
3. 獲取 HuggingFace Token:
   - 前往 Settings → Access Tokens
   - 創建 **Write** 權限的 token
   - 複製並保存（只顯示一次）

### 2. GitHub Repository 設置

#### 添加 GitHub Secrets

在 GitHub Repository 的 Settings → Secrets and variables → Actions 中添加：

| Secret Name | Description | Example |
|------------|-------------|---------|
| `HF_ALCHEMISTAIDEV01` | HuggingFace access token | `hf_xxxxxxxxxx` |

#### 配置 Git LFS

```bash
# 1. 安裝 Git LFS (如果尚未安裝)
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt-get install git-lfs

# Windows
# 從 https://git-lfs.github.com/ 下載安裝

# 2. 初始化 Git LFS
git lfs install

# 3. 將大模型文件添加到 LFS (項目中已有 .gitattributes)
git lfs track "*.pth"
git lfs track "*.bin"
git lfs track "*.onnx"

# 4. 驗證 LFS 配置
git lfs ls-files
```

### 3. 本地文件結構檢查

確保以下文件存在且配置正確：

```
Multi_AOP_FastAPI/
├── .gitattributes          # ✅ Git LFS 配置
├── .github/
│   └── workflows/
│       └── sync_to_hub.yml # ✅ GitHub Actions workflow
├── README.md               # ✅ 包含 HF metadata (前 8 行)
├── docker/
│   └── Dockerfile          # ✅ 監聽 7860 端口
├── app/                    # FastAPI 應用代碼
└── predict/
    └── model/
        └── best_model_Oct13.pth  # 大模型文件 (將由 Git LFS 管理)
```

---

## 🚀 部署步驟

### 方法一：使用 GitHub Actions (推薦)

#### Step 1: 將模型文件遷移到 Git LFS

⚠️ **重要**：如果模型文件已經在 Git 歷史中，需要先清理：

```bash
# 1. 創建新分支進行 LFS 遷移
git checkout -b lfs-migration

# 2. 從歷史中移除大文件（但保留在工作目錄）
git filter-repo --path predict/model/best_model_Oct13.pth --invert-paths

# 3. 重新添加為 LFS 文件
git lfs track "*.pth"
git add .gitattributes
git add predict/model/best_model_Oct13.pth
git commit -m "chore: migrate model files to Git LFS"

# 4. 合併到 production 分支
git checkout production
git merge lfs-migration --allow-unrelated-histories

# 5. 強制推送（因為歷史已改寫）
git push origin production --force
```

#### Step 2: 觸發自動部署

```bash
# 推送到 production 分支會自動觸發 GitHub Actions
git push origin production
```

#### Step 3: 監控部署進度

1. 前往 GitHub Repository → Actions 標籤
2. 查看 "Sync to Hugging Face Hub (Production)" workflow
3. 等待 workflow 完成（通常 5-10 分鐘）
4. 前往 HuggingFace Space 查看構建日誌

### 方法二：手動部署

```bash
# 1. 添加 HuggingFace Space 為遠端
git remote add hf-space https://huggingface.co/spaces/AlchemistAIDev01/Multi_AOP_FastAPI

# 2. 推送到 HuggingFace
git push hf-space production:main

# 注意：需要先設置 HuggingFace CLI 認證
huggingface-cli login
```

---

## 🔍 問題排查

### 問題 1: Configuration Error - Missing configuration in README

**錯誤訊息**：
```
Configuration error
Missing configuration in README
```

**原因**：README.md 缺少 HuggingFace Space 所需的 YAML front matter。

**解決方案**：
確保 README.md 開頭包含以下內容：

```yaml
---
title: Multi AOP FastAPI
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---
```

### 問題 2: Docker Build 失敗 - Port 不匹配

**錯誤訊息**：
```
Application failed to respond on port 7860
```

**原因**：Dockerfile 監聽的端口與 README.md 中的 `app_port` 不匹配。

**解決方案**：
確保 Dockerfile 的 CMD 使用正確的端口：

```dockerfile
CMD ["conda", "run", "-n", "app", "sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
```

### 問題 3: 模型文件過大無法推送

**錯誤訊息**：
```
remote: error: File predict/model/best_model_Oct13.pth is 123.45 MB; this exceeds GitHub's file size limit of 100.00 MB
```

**原因**：模型文件超過 Git 的單文件大小限制（100 MB）。

**解決方案**：
使用 Git LFS：

```bash
# 1. 追蹤大文件
git lfs track "predict/model/*.pth"

# 2. 添加 .gitattributes
git add .gitattributes

# 3. 重新添加模型文件
git rm --cached predict/model/best_model_Oct13.pth
git add predict/model/best_model_Oct13.pth
git commit -m "chore: use Git LFS for model files"
git push
```

### 問題 4: Docker Build 找不到模型文件

**錯誤訊息**：
```
COPY failed: file not found in build context
```

**原因**：GitHub Workflow 使用 `git filter-branch` 刪除了模型文件。

**解決方案**：
移除 workflow 中的 `git filter-branch` 步驟，改用 Git LFS（已在更新的 workflow 中修正）。

### 問題 5: LFS 文件未正確上傳

**檢查方法**：
```bash
# 查看 LFS 追蹤的文件
git lfs ls-files

# 應該看到類似輸出：
# 1a2b3c4d5e * predict/model/best_model_Oct13.pth
```

**解決方案**：
```bash
# 確保 LFS 已安裝
git lfs install

# 重新追蹤並提交
git lfs track "*.pth"
git add .gitattributes
git add predict/model/best_model_Oct13.pth
git commit -m "fix: ensure model files are tracked by LFS"
git push
```

---

## 🔧 技術細節

### 端口配置邏輯

Dockerfile 使用環境變量 `PORT` 來支持多平台部署：

| 平台 | 默認端口 | 配置方式 |
|------|---------|---------|
| HuggingFace Spaces | 7860 | 自動設置（通過 README.md） |
| Render | 10000 | 通過 `PORT` 環境變量 |
| 本地開發 | 7860 | 默認值或自定義 `PORT` |

### Git LFS 工作原理

1. Git LFS 將大文件內容存儲在單獨的 LFS 服務器
2. Git 倉庫只存儲指向大文件的**指針文件**（~100 bytes）
3. 克隆倉庫時，Git LFS 自動下載實際文件內容

### HuggingFace Space 構建流程

```
1. GitHub Actions 推送代碼到 HF Space
   ↓
2. HF 檢測到更新，觸發自動構建
   ↓
3. 讀取 README.md 中的 metadata (sdk: docker, app_port: 7860)
   ↓
4. 使用 Docker 構建 image (執行 Dockerfile)
   ↓
5. 啟動容器，映射端口 7860
   ↓
6. 健康檢查 (HEALTHCHECK in Dockerfile)
   ↓
7. 部署成功，公開 API endpoint
```

---

## 🧪 本地測試

### 測試 Docker 構建

```bash
# 1. 進入 docker 目錄
cd docker

# 2. 構建 image (使用 HF 配置)
docker build -f Dockerfile -t multi-aop-hf:latest ../

# 3. 運行容器（模擬 HF Spaces）
docker run -p 7860:7860 multi-aop-hf:latest

# 4. 測試 API
curl http://localhost:7860/health
curl http://localhost:7860/docs
```

### 測試 API Endpoints

```bash
# Health check
curl -X GET http://localhost:7860/health

# API documentation (Swagger UI)
open http://localhost:7860/docs

# 單個序列預測
curl -X POST http://localhost:7860/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ACDEFGHIKLMNPQRSTVWY"}'

# 批次預測
curl -X POST http://localhost:7860/api/v1/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": [
      "ACDEFGHIKLMNPQRSTVWY",
      "MKTIIALSYIFCLVFA"
    ]
  }'
```

---

## 📊 部署檢查清單

使用此清單確保所有配置正確：

- [ ] **README.md** 包含正確的 YAML front matter
  - [ ] `sdk: docker`
  - [ ] `app_port: 7860`
- [ ] **.gitattributes** 配置 Git LFS
  - [ ] `*.pth filter=lfs diff=lfs merge=lfs -text`
- [ ] **Dockerfile** 配置正確
  - [ ] `EXPOSE 7860`
  - [ ] `CMD` 使用 `--port ${PORT:-7860}`
- [ ] **GitHub Secrets** 已設置
  - [ ] `HF_ALCHEMISTAIDEV01` (HuggingFace token)
- [ ] **模型文件** 已遷移到 Git LFS
  - [ ] `git lfs ls-files` 顯示 .pth 文件
- [ ] **GitHub Workflow** 已更新
  - [ ] 移除 `git filter-branch` 步驟
  - [ ] 啟用 `lfs: true`

---

## 🔗 相關資源

- [HuggingFace Spaces 官方文檔](https://huggingface.co/docs/hub/spaces)
- [Docker Spaces 配置參考](https://huggingface.co/docs/hub/spaces-config-reference)
- [Git LFS 官方文檔](https://git-lfs.github.com/)
- [GitHub Actions 文檔](https://docs.github.com/en/actions)

---

## 🎉 部署成功後

部署成功後，您的 API 將在以下地址可用：

```
https://alchemistaidev01-multi-aop-fastapi.hf.space
```

API 文檔：
```
https://alchemistaidev01-multi-aop-fastapi.hf.space/docs
```

您可以在 HuggingFace Space 的頁面上查看：
- 實時日誌
- 資源使用情況
- API 狀態
- 訪問統計

---

## ⚠️ 注意事項

1. **免費版限制**：
   - CPU: 2 vCPU
   - RAM: 16 GB
   - 無 GPU
   - 如需 GPU，需升級到付費版

2. **休眠機制**：
   - 48 小時無訪問後自動休眠
   - 下次訪問時自動喚醒（需等待 1-2 分鐘）

3. **模型大小**：
   - 單個 LFS 文件最大 5 GB
   - 總倉庫大小建議不超過 50 GB

4. **安全建議**：
   - 不要在代碼中硬編碼 token
   - 定期更新 HuggingFace access token
   - 使用環境變量管理敏感信息

---

## 📧 支持

如遇到問題，請：
1. 查看 HuggingFace Space 的構建日誌
2. 檢查 GitHub Actions 的 workflow 日誌
3. 參考本指南的「問題排查」章節
4. 聯繫項目維護者

---

*最後更新: 2024年12月*


