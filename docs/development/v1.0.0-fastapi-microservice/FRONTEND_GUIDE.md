# 前端對接指南（MVP）

適用對象：前端工程師、產品經理  
版本：v1.0.0（MVP/Demo）

---

## 1. 產品與功能概覽
- **目的**：提供抗氧化肽（AOP）預測 API，輸入氨基酸序列，輸出是否為 AOP 的機率與判定。
- **核心端點**：
  - `POST /api/v1/predict/single`：單序列預測
  - `POST /api/v1/predict/batch`：批次預測（最多 100 條）
  - `GET  /api/v1/model/info`：模型資訊
  - `GET  /health`：健康檢查
- **輸入限制**：序列長度 2~50，僅允許 20 種標準氨基酸字元（A,R,N,D,C,E,Q,G,H,I,L,K,M,F,P,S,T,W,Y,V）。
- **輸出重點**：`prediction` (0/1)、`probability` (0-1)、`confidence` (low/medium/high)、`is_aop` (bool)。

---

## 2. 環境與網域
請向後端確認部署環境的 Base URL，例如：
- 生產 / Demo：`https://<your-domain>/`
- 健康檢查：`GET /health`
- API 前綴：`/api/v1/...`

前端需在 `.env` 設定變數（示例）：
```
NEXT_PUBLIC_API_BASE=https://<your-domain>
```

---

## 3. API 詳解與範例

### 3.1 單序列預測
- **Endpoint**：`POST /api/v1/predict/single`
- **Request Body**：
```json
{ "sequence": "MKLLVVVFCLVLAAP" }
```
- **Response 200**：
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
- **錯誤（422）**：輸入不合法（長度、字元、空字串）。

### 3.2 批次預測
- **Endpoint**：`POST /api/v1/predict/batch`
- **Request Body**（最多 100 條）：
```json
{
  "sequences": [
    "MKLLVVVFCLVLAAP",
    "ACDEFGHIKLMNPQRSTVWY"
  ]
}
```
- **Response 200**：
```json
{
  "total": 2,
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
    }
  ],
  "processing_time_seconds": 2.5
}
```
- **錯誤（422）**：空陣列、超過 100 條、無效序列。

### 3.3 模型資訊
- **Endpoint**：`GET /api/v1/model/info`
- **Response 200**：
```json
{
  "model_version": "1.0.0",
  "model_path": "predict/model/best_model_Oct13.pth",
  "device": "cpu",
  "seq_length": 50,
  "loaded_at": "2024-12-19T10:00:00Z",
  "is_loaded": true
}
```

### 3.4 健康檢查
- **Endpoint**：`GET /health`
- **Response 200**：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-12-19T10:00:00Z",
  "environment": "production",
  "message": ""
}
```

---

## 4. 錯誤模型與前端處理建議
- 統一錯誤格式（`ErrorResponse`）：
```json
{
  "detail": "Sequence length must be between 2 and 50 amino acids",
  "error_type": "ValidationError",
  "status_code": 422
}
```
- 常見情境：
  - 422：輸入驗證失敗（長度、非法字元、空陣列）。
  - 503：模型未加載（冷啟動或資源不足）。
  - 500：非預期錯誤。
- 前端建議：
  - 表單驗證：長度 2~50，僅 20 種字元，批次數量 ≤100。
  - 錯誤提示：針對 422 顯示具體原因；503/500 顯示重試或聯繫支援。
  - 重試策略：503/500 可提示稍後重試；422 不要自動重試。

---

## 5. 前端對接範例（fetch）
```ts
// 單序列
const base = process.env.NEXT_PUBLIC_API_BASE;
const res = await fetch(`${base}/api/v1/predict/single`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ sequence: 'MKLLVVVFCLVLAAP' }),
});
const data = await res.json();

// 批次
await fetch(`${base}/api/v1/predict/batch`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ sequences: ['SEQ1', 'SEQ2'] }),
});
```

---

## 6. UX / 產品建議
1. **輸入驗證即時提示**：長度、字元集、批次上限。
2. **批次結果呈現**：表格 + 篩選（prediction/信心度），支援導出 CSV。
3. **狀態反饋**：顯示 loading/processing_time，錯誤提示明確。
4. **健康狀態顯示**：在頁面顯示 `/health` 結果（模型是否已載入）。
5. **示例與引導**：預填範例序列、一鍵測試、API 範例代碼區塊。
6. **錯誤指引**：對 422 顯示具體字段問題；503 建議稍後重試。

---

## 7. 前端集成清單（Checklist）
- [ ] 設定 `NEXT_PUBLIC_API_BASE` 指向後端域名
- [ ] 表單驗證（2~50 長度、20 種字元、批次 ≤100）
- [ ] 錯誤處理（422/503/500 分流）
- [ ] 結果表格與排序/篩選
- [ ] 顯示 processing time 與概率/信心度
- [ ] 健康檢查指示燈（模型載入狀態）
- [ ] 提供範例序列與一鍵測試

---

## 8. 給前端的快速起步
1. 設定環境變數 `NEXT_PUBLIC_API_BASE`
2. 實作兩個呼叫：
   - 單序列：`POST /api/v1/predict/single`
   - 批次：`POST /api/v1/predict/batch`
3. 建立結果表格與錯誤提示
4. 加入健康檢查顯示（可定期輪詢 `/health`）

---

## 9. 問題聯繫
- 後端 API/模型問題：請聯繫後端工程師
- 產品需求與流程：請聯繫產品經理

--- 

（本指南聚焦前端對接與 UX 建議；API 詳細規格已在本文提供，可直接用於對接。） 


