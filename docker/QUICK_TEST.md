# 快速測試指南

本文檔提供快速測試 Docker 容器化應用程式的步驟。

## 🚀 快速開始

### 1. 啟動服務

```bash
# 進入 docker 目錄
cd docker

# 使用 docker-compose 啟動服務
docker-compose up -d

# 或使用 Makefile
make up
```

### 2. 檢查服務狀態

```bash
# 檢查容器是否運行
docker ps | grep multi-aop-api

# 查看日誌
docker-compose logs -f

# 或使用 Makefile
make logs
```

### 3. 運行自動化測試

```bash
# 運行完整測試套件
./test.sh

# 或使用 Makefile
make test-full
```

## 📋 手動測試步驟

### 步驟 1: 健康檢查

```bash
curl http://localhost:8000/health | jq .
```

**預期結果**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-12-19T...",
  "environment": "production"
}
```

### 步驟 2: 模型信息

```bash
curl http://localhost:8000/api/v1/model/info | jq .
```

**預期結果**:
```json
{
  "model_version": "1.0.0",
  "model_path": "predict/model/best_model_Oct13.pth",
  "device": "cpu",
  "seq_length": 50,
  "loaded_at": "2024-12-19T...",
  "is_loaded": true
}
```

### 步驟 3: 單個序列預測

```bash
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKLLVVVFCLVLAAP"}' | jq .
```

**預期結果**:
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

### 步驟 4: 批次預測

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": [
      "MKLLVVVFCLVLAAP",
      "ACDEFGHIKLMNPQRSTVWY",
      "TTTTTTTTTTTTTTTTTTTT"
    ]
  }' | jq .
```

**預期結果**:
```json
{
  "total": 3,
  "results": [
    {
      "sequence": "MKLLVVVFCLVLAAP",
      "prediction": 1,
      "probability": 0.85,
      "confidence": "high",
      "is_aop": true
    },
    ...
  ],
  "processing_time_seconds": 2.5
}
```

### 步驟 5: 錯誤場景測試

```bash
# 測試序列太短
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{"sequence": "A"}' | jq .

# 測試序列太長
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{"sequence": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}' | jq .

# 測試無效字符
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}' | jq .
```

**預期結果**: 所有請求都應該返回 HTTP 422 錯誤

## ✅ 測試檢查清單

### 基本功能
- [ ] 容器成功啟動
- [ ] 健康檢查返回 200 OK
- [ ] 模型信息端點正常
- [ ] 單個序列預測成功
- [ ] 批次預測成功

### 錯誤處理
- [ ] 序列太短返回 422
- [ ] 序列太長返回 422
- [ ] 無效字符返回 422
- [ ] 空批次返回 422
- [ ] 缺少字段返回 422

### 性能
- [ ] 單個序列響應時間 < 5 秒
- [ ] 批次預測（10個序列）響應時間 < 30 秒
- [ ] 無明顯內存洩漏

## 🐛 常見問題

### 問題 1: 容器無法啟動

```bash
# 檢查日誌
docker-compose logs

# 檢查端口是否被佔用
lsof -i :8000

# 重新構建並啟動
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### 問題 2: 模型未加載

```bash
# 檢查模型文件是否存在
ls -la ../predict/model/best_model_Oct13.pth

# 檢查容器內的模型路徑
docker exec multi-aop-api ls -la /app/predict/model/

# 檢查環境變量
docker exec multi-aop-api env | grep MODEL_PATH
```

### 問題 3: API 無響應

```bash
# 檢查容器狀態
docker ps -a | grep multi-aop-api

# 檢查日誌中的錯誤
docker logs multi-aop-api | grep -i error

# 重啟容器
docker-compose restart
```

## 📊 性能基準

在 CPU 模式下，預期性能：

- **單個序列預測**: 1-3 秒
- **批次預測 (10個序列)**: 5-15 秒
- **批次預測 (100個序列)**: 30-60 秒
- **內存使用**: 2-4 GB
- **CPU 使用率**: 50-100% (預測時)

## 🔧 進階測試

### 壓力測試

```bash
# 使用 Apache Bench 進行壓力測試
ab -n 100 -c 10 -p test_data.json -T application/json \
   http://localhost:8000/api/v1/predict/single
```

### 並發測試

```bash
# 使用 parallel 進行並發測試
seq 1 10 | parallel -j 10 \
  'curl -X POST http://localhost:8000/api/v1/predict/single \
   -H "Content-Type: application/json" \
   -d "{\"sequence\": \"MKLLVVVFCLVLAAP\"}"'
```

### 監控資源使用

```bash
# 監控容器資源使用
docker stats multi-aop-api

# 查看詳細日誌
docker logs -f multi-aop-api
```

## 📝 測試報告模板

```markdown
## 測試報告

**日期**: YYYY-MM-DD
**環境**: Docker (CPU/GPU)
**版本**: v1.0.0

### 測試結果
- [ ] 基本功能測試: 通過/失敗
- [ ] 錯誤處理測試: 通過/失敗
- [ ] 性能測試: 通過/失敗

### 發現的問題
1. [問題描述]
2. [問題描述]

### 備註
[其他備註]
```

---

**最後更新**: 2024-12-19  
**版本**: v1.0.0

