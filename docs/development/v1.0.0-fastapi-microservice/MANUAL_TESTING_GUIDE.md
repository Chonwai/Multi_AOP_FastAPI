# 手動測試指南

本文檔提供使用 curl 命令手動測試 Multi-AOP FastAPI 微服務的完整指南。

---

## 📋 目錄

1. [前置準備](#前置準備)
2. [健康檢查和模型信息](#健康檢查和模型信息)
3. [單個序列預測](#單個序列預測)
4. [批次預測](#批次預測)
5. [錯誤場景測試](#錯誤場景測試)
6. [故障排查](#故障排查)

---

## 🔧 前置準備

### 1. 確保服務運行

```bash
# 檢查服務是否運行
curl http://localhost:8000/health

# 預期響應
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-12-19T10:00:00"
}
```

### 2. 設置基礎URL（可選）

```bash
# 設置環境變量
export API_URL="http://localhost:8000"

# 或直接使用完整URL
```

---

## 🏥 健康檢查和模型信息

### 健康檢查端點

```bash
# GET /health
curl -X GET http://localhost:8000/health

# 預期響應（200 OK）
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-12-19T10:00:00"
}

# 如果模型未加載
{
  "status": "unhealthy",
  "model_loaded": false,
  "message": "Model not loaded",
  "timestamp": "2024-12-19T10:00:00"
}
```

### 模型信息端點

```bash
# GET /api/v1/model/info
curl -X GET http://localhost:8000/api/v1/model/info

# 預期響應（200 OK）
{
  "model_version": "1.0.0",
  "model_path": "/app/models/best_model.pth",
  "device": "cpu",
  "seq_length": 50,
  "loaded_at": "2024-12-19T09:55:00"
}
```

---

## 🔬 單個序列預測

### 正常預測請求

```bash
# POST /api/v1/predict/single
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MKLLVVVFCLVLAAP"
  }'

# 預期響應（200 OK）
{
  "sequence": "MKLLVVVFCLVLAAP",
  "prediction": 1,
  "probability": 0.85,
  "confidence": "high",
  "is_aop": true,
  "message": "Predicted as AOP"
}
```

### 更多示例序列

```bash
# 示例1：短序列
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ACDEFGHIKLMNPQRSTVWY"}'

# 示例2：中等長度序列
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKLLVVVFCLVLAAPTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"}'

# 示例3：已知AOP序列（如果有的話）
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{"sequence": "YOUR_KNOWN_AOP_SEQUENCE_HERE"}'
```

---

## 📦 批次預測

### 正常批次預測請求

```bash
# POST /api/v1/predict/batch
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": [
      "MKLLVVVFCLVLAAP",
      "ACDEFGHIKLMNPQRSTVWY",
      "TTTTTTTTTTTTTTTTTTTT"
    ]
  }'

# 預期響應（200 OK）
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
    {
      "sequence": "ACDEFGHIKLMNPQRSTVWY",
      "prediction": 0,
      "probability": 0.23,
      "confidence": "low",
      "is_aop": false
    },
    {
      "sequence": "TTTTTTTTTTTTTTTTTTTT",
      "prediction": 0,
      "probability": 0.15,
      "confidence": "low",
      "is_aop": false
    }
  ],
  "processing_time_seconds": 2.5
}
```

### 大批次測試（接近上限）

```bash
# 測試最大批次大小（100個序列）
# 注意：這裡只是示例，實際需要準備100個序列
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": [
      "SEQUENCE1",
      "SEQUENCE2",
      ...
      "SEQUENCE100"
    ]
  }'
```

---

## ❌ 錯誤場景測試

### 1. 無效的序列長度

```bash
# 序列太短（少於2個氨基酸）
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{"sequence": "A"}'

# 預期響應（422 Unprocessable Entity）
{
  "detail": [
    {
      "loc": ["body", "sequence"],
      "msg": "Sequence length must be between 2 and 50 amino acids",
      "type": "value_error"
    }
  ]
}

# 序列太長（超過50個氨基酸）
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{"sequence": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"}'

# 預期響應（422 Unprocessable Entity）
{
  "detail": [
    {
      "loc": ["body", "sequence"],
      "msg": "Sequence length must be between 2 and 50 amino acids",
      "type": "value_error"
    }
  ]
}
```

### 2. 無效的氨基酸字符

```bash
# 包含非標準氨基酸字符
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}'

# 預期響應（422 Unprocessable Entity）
{
  "detail": [
    {
      "loc": ["body", "sequence"],
      "msg": "Sequence contains invalid amino acid characters. Only standard 20 amino acids are allowed.",
      "type": "value_error"
    }
  ]
}
```

### 3. 缺少必需字段

```bash
# 缺少sequence字段
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{}'

# 預期響應（422 Unprocessable Entity）
{
  "detail": [
    {
      "loc": ["body", "sequence"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 4. 批次大小超限

```bash
# 批次大小超過100
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": ["SEQ1", "SEQ2", ... , "SEQ101"]
  }'

# 預期響應（422 Unprocessable Entity）
{
  "detail": [
    {
      "loc": ["body", "sequences"],
      "msg": "Batch size cannot exceed 100 sequences",
      "type": "value_error"
    }
  ]
}
```

### 5. 空批次

```bash
# 空序列列表
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"sequences": []}'

# 預期響應（422 Unprocessable Entity）
{
  "detail": [
    {
      "loc": ["body", "sequences"],
      "msg": "Batch cannot be empty",
      "type": "value_error"
    }
  ]
}
```

### 6. 無效的JSON格式

```bash
# 無效的JSON
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{invalid json}'

# 預期響應（422 Unprocessable Entity）
{
  "detail": "Invalid JSON format"
}
```

### 7. 模型未加載

```bash
# 如果模型未加載（服務剛啟動或加載失敗）
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKLLVVVFCLVLAAP"}'

# 預期響應（503 Service Unavailable）
{
  "detail": "Model not loaded. Please wait for model initialization or check server logs."
}
```

---

## 🔍 測試場景清單

### 正常場景

- [ ] 健康檢查端點返回200
- [ ] 模型信息端點返回模型詳情
- [ ] 單個序列預測成功（短序列）
- [ ] 單個序列預測成功（中等長度序列）
- [ ] 單個序列預測成功（長序列，接近50）
- [ ] 批次預測成功（小批次，1-10個序列）
- [ ] 批次預測成功（中等批次，10-50個序列）
- [ ] 批次預測成功（大批次，50-100個序列）

### 錯誤場景

- [ ] 序列太短（< 2個氨基酸）返回422
- [ ] 序列太長（> 50個氨基酸）返回422
- [ ] 無效氨基酸字符返回422
- [ ] 缺少必需字段返回422
- [ ] 批次大小超限返回422
- [ ] 空批次返回422
- [ ] 無效JSON返回422
- [ ] 模型未加載返回503

### 性能驗證

- [ ] 單個序列響應時間 < 2秒
- [ ] 批次預測（100個序列）響應時間 < 30秒
- [ ] 並發請求（5-10個）正常處理

---

## 🐛 故障排查

### 問題1：連接被拒絕

```bash
# 錯誤：curl: (7) Failed to connect to localhost port 8000: Connection refused

# 解決方案：
# 1. 檢查服務是否運行
docker ps

# 2. 檢查端口是否正確
docker port <container_name>

# 3. 檢查日誌
docker logs <container_name>
```

### 問題2：模型加載失敗

```bash
# 錯誤：健康檢查返回 model_loaded: false

# 解決方案：
# 1. 檢查模型文件是否存在
docker exec <container_name> ls -la /app/models/

# 2. 檢查模型路徑配置
docker exec <container_name> env | grep MODEL_PATH

# 3. 查看詳細錯誤日誌
docker logs <container_name> | grep -i error
```

### 問題3：預測結果異常

```bash
# 問題：預測概率始終為0或1

# 解決方案：
# 1. 檢查模型是否正確加載
curl http://localhost:8000/api/v1/model/info

# 2. 檢查輸入序列格式
# 確保序列只包含標準20種氨基酸

# 3. 檢查模型文件完整性
docker exec <container_name> file /app/models/best_model.pth
```

### 問題4：響應時間過長

```bash
# 問題：單個序列預測超過2秒

# 解決方案：
# 1. 檢查是否使用GPU（如果配置了）
curl http://localhost:8000/api/v1/model/info | grep device

# 2. 檢查系統資源使用
docker stats <container_name>

# 3. 檢查批次大小是否過大
# 減少批次大小或增加資源限制
```

### 問題5：內存不足

```bash
# 錯誤：容器因內存不足而重啟

# 解決方案：
# 1. 檢查內存使用
docker stats <container_name>

# 2. 減少批次大小限制
# 修改配置中的 MAX_BATCH_SIZE

# 3. 增加Docker內存限制
# 在docker-compose.yml中設置mem_limit
```

---

## 📊 測試結果記錄模板

```markdown
## 測試日期：YYYY-MM-DD

### 環境信息
- Docker版本：x.x.x
- 服務版本：v1.0.0
- 模型路徑：/app/models/best_model.pth
- 設備：CPU/GPU

### 測試結果

#### 健康檢查
- [ ] GET /health - 通過/失敗
- [ ] GET /api/v1/model/info - 通過/失敗

#### 單個預測
- [ ] 正常序列 - 通過/失敗
- [ ] 短序列錯誤 - 通過/失敗
- [ ] 長序列錯誤 - 通過/失敗
- [ ] 無效字符錯誤 - 通過/失敗

#### 批次預測
- [ ] 小批次（10個） - 通過/失敗
- [ ] 中等批次（50個） - 通過/失敗
- [ ] 大批次（100個） - 通過/失敗
- [ ] 批次超限錯誤 - 通過/失敗

#### 性能
- [ ] 單個序列響應時間：X秒
- [ ] 批次預測（100個）響應時間：X秒

### 發現的問題
1. [問題描述]
2. [問題描述]

### 備註
[其他備註]
```

---

## 💡 提示和最佳實踐

1. **使用jq格式化JSON響應**：
   ```bash
   curl ... | jq .
   ```

2. **保存響應到文件**：
   ```bash
   curl ... > response.json
   ```

3. **顯示詳細信息**：
   ```bash
   curl -v ...  # 顯示請求頭和響應頭
   ```

4. **測試並發請求**：
   ```bash
   # 使用parallel或xargs
   seq 1 10 | xargs -P 10 -I {} curl -X POST ...
   ```

5. **監控響應時間**：
   ```bash
   time curl -X POST ...
   ```

---

**最後更新**：2024-12-19  
**版本**：v1.0.0-MVP

