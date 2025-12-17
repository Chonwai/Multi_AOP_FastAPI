# 🚀 HuggingFace 部署快速修復指南

## 問題摘要

您的 Multi-AOP FastAPI 項目在部署到 HuggingFace Spaces 時遇到 "Configuration error"。

## 🔍 發現的問題

| # | 問題 | 嚴重性 | 狀態 |
|---|------|--------|------|
| 1 | 端口不匹配 (README: 7860 vs Dockerfile: 8000) | 🔴 高 | ✅ 已修復 |
| 2 | 缺少 Git LFS 配置 (.gitattributes) | 🔴 高 | ✅ 已修復 |
| 3 | GitHub Workflow 刪除模型文件但 Dockerfile 需要 | 🔴 高 | ✅ 已修復 |

## ✅ 已完成的修復

### 1. 創建 `.gitattributes` (Git LFS 配置)

新文件已創建，配置所有大文件格式使用 Git LFS：
- `*.pth` (PyTorch 模型)
- `*.bin` (二進制模型)
- `*.onnx` (ONNX 模型)

### 2. 修改 `docker/Dockerfile`

**更改內容**：
- ✅ `EXPOSE 8000` → `EXPOSE 7860`
- ✅ `CMD` 默認端口從 `8000` → `7860`
- ✅ 更新健康檢查使用 `PORT` 環境變量

### 3. 修改 `.github/workflows/sync_to_hub.yml`

**更改內容**：
- ✅ 移除 `git filter-branch` 刪除模型文件的步驟
- ✅ 添加 Git LFS 設置步驟
- ✅ 確保 `lfs: true` 在 checkout 步驟中啟用

### 4. 創建詳細部署指南

新文件：`docs/HUGGINGFACE_DEPLOYMENT_GUIDE.md`
- 完整的部署步驟
- 問題排查指南
- 本地測試方法
- 技術細節說明

## 🚦 接下來的步驟

### ⚠️ 重要：需要手動執行的操作

由於模型文件可能已經在 Git 歷史中（非 LFS），您需要執行以下步驟：

#### Step 1: 安裝 Git LFS

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt-get install git-lfs

# 初始化 Git LFS
git lfs install
```

#### Step 2: 將現有模型文件遷移到 LFS

```bash
# 1. 確認當前在主分支
git checkout main  # 或 production

# 2. 查看模型文件是否已在 LFS 中
git lfs ls-files

# 3. 如果沒有看到 .pth 文件，執行遷移
# (這會將文件從普通 Git 轉換為 LFS)
git rm --cached predict/model/best_model_Oct13.pth
git add predict/model/best_model_Oct13.pth

# 4. 提交更改
git commit -m "chore: migrate model files to Git LFS"

# 5. 推送到遠端
git push origin main  # 或 production
```

#### Step 3: 觸發部署

```bash
# 如果您使用 production 分支部署
git checkout production
git merge main
git push origin production

# GitHub Actions 會自動觸發並推送到 HuggingFace
```

#### Step 4: 驗證部署

1. 前往 GitHub → Actions 查看 workflow 狀態
2. 前往 HuggingFace Space 查看構建日誌：
   ```
   https://huggingface.co/spaces/AlchemistAIDev01/Multi_AOP_FastAPI
   ```
3. 等待構建完成（約 5-10 分鐘）
4. 測試 API：
   ```bash
   curl https://alchemistaidev01-multi-aop-fastapi.hf.space/health
   ```

## 🧪 本地測試（可選但推薦）

在推送到 HuggingFace 之前，建議先在本地測試：

```bash
# 1. 構建 Docker image
cd docker
docker build -f Dockerfile -t multi-aop-test:latest ../

# 2. 運行容器（使用 7860 端口）
docker run -p 7860:7860 multi-aop-test:latest

# 3. 在另一個終端測試
curl http://localhost:7860/health
curl http://localhost:7860/docs

# 4. 測試預測 API
curl -X POST http://localhost:7860/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ACDEFGHIKLMNPQRSTVWY"}'
```

## 📋 檢查清單

在推送之前，請確認：

- [ ] 已安裝 Git LFS (`git lfs version`)
- [ ] `.gitattributes` 文件存在
- [ ] 模型文件已遷移到 LFS (`git lfs ls-files` 顯示 .pth)
- [ ] `docker/Dockerfile` 使用端口 7860
- [ ] `.github/workflows/sync_to_hub.yml` 已更新
- [ ] `README.md` 開頭包含正確的 YAML front matter
- [ ] (可選) 本地 Docker 測試通過

## 🔄 如果部署失敗

### 查看日誌

1. **GitHub Actions 日誌**：
   ```
   https://github.com/[your-username]/Multi_AOP_FastAPI/actions
   ```

2. **HuggingFace Space 日誌**：
   - 前往 Space 頁面
   - 點擊 "Logs" 標籤
   - 查看構建和運行日誌

### 常見錯誤

| 錯誤訊息 | 可能原因 | 解決方案 |
|---------|---------|---------|
| "Configuration error" | README.md 缺少 metadata | 確保前 8 行包含 YAML |
| "Port 7860 not responding" | Dockerfile 端口錯誤 | 檢查 EXPOSE 和 CMD |
| "File not found: *.pth" | LFS 未正確配置 | 執行 Step 2 遷移步驟 |
| "Authentication failed" | GitHub Secret 錯誤 | 檢查 HF token |

## 📚 延伸閱讀

- 完整部署指南：`docs/HUGGINGFACE_DEPLOYMENT_GUIDE.md`
- HuggingFace Spaces 文檔：https://huggingface.co/docs/hub/spaces
- Git LFS 教程：https://git-lfs.github.com/

## 💡 技術亮點

您的項目已經包含了優秀的設計模式：

1. **單例模式** (Singleton Pattern)
   - `ModelManager`: 線程安全的模型管理
   - `Settings`: 統一配置管理

2. **依賴注入** (Dependency Injection)
   - `PredictionService` 接受可選的 `ModelManager`

3. **工廠模式** (Factory Pattern)
   - `create_in_memory_loader`: DataLoader 工廠

這些都是業界標準的最佳實踐！👏

## 🎯 預期結果

完成上述步驟後，您應該能夠：
- ✅ 成功部署到 HuggingFace Spaces
- ✅ 通過公開 URL 訪問 API
- ✅ 查看 Swagger 文檔 (`/docs`)
- ✅ 執行肽段預測

## 🆘 需要幫助？

如果遇到問題：
1. 檢查本文件的「常見錯誤」章節
2. 閱讀完整部署指南
3. 查看 GitHub Actions 和 HuggingFace 的日誌

---

**祝您部署順利！** 🚀


