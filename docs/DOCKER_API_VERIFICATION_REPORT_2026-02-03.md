# Docker 化微服務 API 驗證報告（CPU）

**日期**：2026-02-03  
**版本**：Multi-AOP FastAPI（Dockerized）  
**環境**：CPU-only（成本控制）  
**目標**：驗證 Docker 化 microservice 的可用性與 API 正常性

---

## 1. 驗證範圍與方法（第一性原理）

**核心假設**：若容器能成功建置、啟動，且關鍵 API 端點在 CPU 模式下可正常回應，即可判定 Docker 化 microservice 可用。  

**驗證範圍**：
1. Docker image build
2. Container 啟動與健康檢查
3. API 端點（health、model info、single、batch）

**依據**：
- [docker/TESTING.md](../docker/TESTING.md)

---

## 2. 建置與啟動結果

### 2.1 Docker Build
**結果**：✅ 成功  
**映像名稱**：`multi-aop-api:latest`

### 2.2 Container Run
**結果**：✅ 成功  
**模式**：CPU  
**容器名稱**：`multi-aop-api`  

啟動指令（參考）：
```
docker run -d \
  --name multi-aop-api \
  -p 8000:8000 \
  -e MODEL_PATH=predict/model/best_model_Oct13.pth \
  -e DEVICE=cpu \
  -v $(pwd)/predict/model:/app/predict/model:ro \
  multi-aop-api:latest
```

---

## 3. API 驗證結果

### 3.1 Health Check
**端點**：`GET /health`  
**結果**：✅ 200 OK  
**要點**：`model_loaded: true`

### 3.2 Model Info
**端點**：`GET /api/v1/model/info`  
**結果**：✅ 200 OK  
**要點**：
- `device: cpu`
- `seq_length: 50`
- `is_loaded: true`

### 3.3 單筆預測
**端點**：`POST /api/v1/predict/single`  
**輸入**：`MKLLVVVFCLVLAAP`  
**結果**：✅ 200 OK  
**輸出摘要**：
- `prediction: 1`
- `probability: 0.7099`
- `confidence: medium`

### 3.4 批次預測
**端點**：`POST /api/v1/predict/batch`  
**輸入**：`["MKLLVVVFCLVLAAP", "ACDEFGHIKLMNPQRSTVWY"]`  
**結果**：✅ 200 OK  
**輸出摘要**：
- 兩筆結果皆成功回傳
- `processing_time_seconds` 正常

---

## 4. 結論（PM/技術主管觀點）

**結論**：Docker 化 microservice 在 CPU-only 環境下**可正常運作**，且主要 API 端點已通過驗證。  

**可交付狀態**：✅ 可用於內部測試/展示/部署

---

## 5. 風險與建議

### 5.1 風險
- CPU 版本仍存在 xLSTM 權重 shape mismatch 跳過情況（已知問題）

### 5.2 建議
- 請 Algorithm Engineer 提供 **CPU 專用 checkpoint** 以完整對齊
- 後續可將此驗證流程自動化加入 CI

---

## 6. 附錄

**測試指令參考**（皆通過）：

- Health:
```
curl http://localhost:8000/health
```

- Model Info:
```
curl http://localhost:8000/api/v1/model/info
```

- Single:
```
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{"sequence":"MKLLVVVFCLVLAAP"}'
```

- Batch:
```
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"sequences":["MKLLVVVFCLVLAAP","ACDEFGHIKLMNPQRSTVWY"]}'
```
