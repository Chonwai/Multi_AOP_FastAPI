#!/bin/bash

# Multi-AOP FastAPI Docker 測試腳本
# 用於驗證 Docker 容器化應用程式的功能

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
API_URL="${API_URL:-http://localhost:8000}"
CONTAINER_NAME="${CONTAINER_NAME:-multi-aop-api}"
TIMEOUT=30

# 測試計數器
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# 輔助函數
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
    ((TESTS_PASSED++))
    ((TESTS_TOTAL++))
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
    ((TESTS_FAILED++))
    ((TESTS_TOTAL++))
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_test() {
    echo -e "\n${BLUE}測試: $1${NC}"
}

# 檢查 HTTP 響應
check_response() {
    local url=$1
    local expected_status=$2
    local description=$3
    
    print_test "$description"
    
    response=$(curl -s -w "\n%{http_code}" "$url" || echo -e "\n000")
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" == "$expected_status" ]; then
        print_success "$description (HTTP $http_code)"
        echo "$body" | jq . 2>/dev/null || echo "$body"
        return 0
    else
        print_error "$description (期望 HTTP $expected_status, 實際 HTTP $http_code)"
        echo "響應: $body"
        return 1
    fi
}

# 檢查 JSON 響應中的字段
check_json_field() {
    local json=$1
    local field=$2
    local expected_value=$3
    local description=$4
    
    actual_value=$(echo "$json" | jq -r ".$field" 2>/dev/null)
    
    if [ "$actual_value" == "$expected_value" ]; then
        print_success "$description: $field = $actual_value"
        return 0
    else
        print_error "$description: 期望 $field = $expected_value, 實際 = $actual_value"
        return 1
    fi
}

# 等待服務就緒
wait_for_service() {
    print_info "等待服務啟動..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "$API_URL/health" > /dev/null 2>&1; then
            print_success "服務已就緒"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    print_error "服務未能在 $((max_attempts * 2)) 秒內啟動"
    return 1
}

# 測試 1: 環境檢查
test_environment() {
    print_header "1. 環境檢查"
    
    # 檢查 Docker
    if command -v docker &> /dev/null; then
        print_success "Docker 已安裝"
    else
        print_error "Docker 未安裝"
        return 1
    fi
    
    # 檢查容器狀態
    if docker ps | grep -q "$CONTAINER_NAME"; then
        print_success "容器 $CONTAINER_NAME 正在運行"
    else
        print_error "容器 $CONTAINER_NAME 未運行"
        print_info "請先運行: docker-compose -f docker/docker-compose.yml up -d"
        return 1
    fi
    
    # 檢查端口
    if curl -s -f "$API_URL/health" > /dev/null 2>&1; then
        print_success "API 端點可訪問 ($API_URL)"
    else
        print_error "API 端點不可訪問 ($API_URL)"
        return 1
    fi
}

# 測試 2: 健康檢查
test_health_check() {
    print_header "2. 健康檢查測試"
    
    response=$(curl -s "$API_URL/health")
    
    if [ $? -eq 0 ]; then
        print_success "健康檢查端點響應正常"
        echo "$response" | jq . 2>/dev/null || echo "$response"
        
        # 檢查模型是否已加載
        model_loaded=$(echo "$response" | jq -r '.model_loaded' 2>/dev/null)
        if [ "$model_loaded" == "true" ]; then
            print_success "模型已成功加載"
        else
            print_error "模型未加載 (model_loaded: $model_loaded)"
            print_info "這可能是正常的，如果模型是延遲加載的"
        fi
    else
        print_error "健康檢查端點無響應"
        return 1
    fi
}

# 測試 3: 模型信息
test_model_info() {
    print_header "3. 模型信息測試"
    
    response=$(curl -s "$API_URL/api/v1/model/info")
    
    if [ $? -eq 0 ]; then
        print_success "模型信息端點響應正常"
        echo "$response" | jq . 2>/dev/null || echo "$response"
        
        # 檢查關鍵字段
        check_json_field "$response" "device" "cpu" "設備類型"
        check_json_field "$response" "seq_length" "50" "序列長度"
    else
        print_error "模型信息端點無響應"
        return 1
    fi
}

# 測試 4: 單個序列預測
test_single_prediction() {
    print_header "4. 單個序列預測測試"
    
    # 測試正常序列
    test_sequence="MKLLVVVFCLVLAAP"
    print_test "測試序列: $test_sequence"
    
    response=$(curl -s -X POST "$API_URL/api/v1/predict/single" \
        -H "Content-Type: application/json" \
        -d "{\"sequence\": \"$test_sequence\"}")
    
    http_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/api/v1/predict/single" \
        -H "Content-Type: application/json" \
        -d "{\"sequence\": \"$test_sequence\"}")
    
    if [ "$http_code" == "200" ]; then
        print_success "單個序列預測成功 (HTTP 200)"
        echo "$response" | jq . 2>/dev/null || echo "$response"
        
        # 檢查響應字段
        sequence=$(echo "$response" | jq -r '.sequence' 2>/dev/null)
        probability=$(echo "$response" | jq -r '.probability' 2>/dev/null)
        
        if [ "$sequence" == "$test_sequence" ]; then
            print_success "響應序列匹配"
        fi
        
        if [ ! -z "$probability" ] && [ "$probability" != "null" ]; then
            print_success "預測概率: $probability"
        fi
    else
        print_error "單個序列預測失敗 (HTTP $http_code)"
        echo "響應: $response"
        return 1
    fi
}

# 測試 5: 批次預測
test_batch_prediction() {
    print_header "5. 批次預測測試"
    
    # 測試小批次
    test_sequences='["MKLLVVVFCLVLAAP", "ACDEFGHIKLMNPQRSTVWY", "TTTTTTTTTTTTTTTTTTTT"]'
    print_test "測試批次: 3 個序列"
    
    response=$(curl -s -X POST "$API_URL/api/v1/predict/batch" \
        -H "Content-Type: application/json" \
        -d "{\"sequences\": $test_sequences}")
    
    http_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/api/v1/predict/batch" \
        -H "Content-Type: application/json" \
        -d "{\"sequences\": $test_sequences}")
    
    if [ "$http_code" == "200" ]; then
        print_success "批次預測成功 (HTTP 200)"
        echo "$response" | jq . 2>/dev/null || echo "$response"
        
        # 檢查結果數量
        total=$(echo "$response" | jq -r '.total' 2>/dev/null)
        if [ "$total" == "3" ]; then
            print_success "批次結果數量正確: $total"
        fi
    else
        print_error "批次預測失敗 (HTTP $http_code)"
        echo "響應: $response"
        return 1
    fi
}

# 測試 6: 錯誤場景測試
test_error_scenarios() {
    print_header "6. 錯誤場景測試"
    
    # 測試 6.1: 序列太短
    print_test "測試: 序列太短 (< 2 個氨基酸)"
    response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/v1/predict/single" \
        -H "Content-Type: application/json" \
        -d '{"sequence": "A"}')
    http_code=$(echo "$response" | tail -n1)
    if [ "$http_code" == "422" ]; then
        print_success "序列太短正確返回 422"
    else
        print_error "序列太短測試失敗 (期望 422, 實際 $http_code)"
    fi
    
    # 測試 6.2: 序列太長
    print_test "測試: 序列太長 (> 50 個氨基酸)"
    long_sequence=$(printf 'A%.0s' {1..51})
    response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/v1/predict/single" \
        -H "Content-Type: application/json" \
        -d "{\"sequence\": \"$long_sequence\"}")
    http_code=$(echo "$response" | tail -n1)
    if [ "$http_code" == "422" ]; then
        print_success "序列太長正確返回 422"
    else
        print_error "序列太長測試失敗 (期望 422, 實際 $http_code)"
    fi
    
    # 測試 6.3: 無效氨基酸字符
    print_test "測試: 無效氨基酸字符"
    response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/v1/predict/single" \
        -H "Content-Type: application/json" \
        -d '{"sequence": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}')
    http_code=$(echo "$response" | tail -n1)
    if [ "$http_code" == "422" ]; then
        print_success "無效字符正確返回 422"
    else
        print_error "無效字符測試失敗 (期望 422, 實際 $http_code)"
    fi
    
    # 測試 6.4: 空批次
    print_test "測試: 空批次"
    response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/v1/predict/batch" \
        -H "Content-Type: application/json" \
        -d '{"sequences": []}')
    http_code=$(echo "$response" | tail -n1)
    if [ "$http_code" == "422" ]; then
        print_success "空批次正確返回 422"
    else
        print_error "空批次測試失敗 (期望 422, 實際 $http_code)"
    fi
    
    # 測試 6.5: 缺少必需字段
    print_test "測試: 缺少必需字段"
    response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/v1/predict/single" \
        -H "Content-Type: application/json" \
        -d '{}')
    http_code=$(echo "$response" | tail -n1)
    if [ "$http_code" == "422" ]; then
        print_success "缺少字段正確返回 422"
    else
        print_error "缺少字段測試失敗 (期望 422, 實際 $http_code)"
    fi
}

# 測試 7: 性能測試
test_performance() {
    print_header "7. 性能測試"
    
    # 單個序列響應時間
    print_test "測試: 單個序列響應時間"
    start_time=$(date +%s.%N)
    curl -s -X POST "$API_URL/api/v1/predict/single" \
        -H "Content-Type: application/json" \
        -d '{"sequence": "MKLLVVVFCLVLAAP"}' > /dev/null
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    
    if (( $(echo "$duration < 5.0" | bc -l) )); then
        print_success "單個序列響應時間: ${duration}秒 (< 5秒)"
    else
        print_error "單個序列響應時間過長: ${duration}秒 (期望 < 5秒)"
    fi
    
    # 批次預測響應時間
    print_test "測試: 批次預測響應時間 (10個序列)"
    test_sequences='["MKLLVVVFCLVLAAP", "ACDEFGHIKLMNPQRSTVWY", "TTTTTTTTTTTTTTTTTTTT", "MKLLVVVFCLVLAAP", "ACDEFGHIKLMNPQRSTVWY", "TTTTTTTTTTTTTTTTTTTT", "MKLLVVVFCLVLAAP", "ACDEFGHIKLMNPQRSTVWY", "TTTTTTTTTTTTTTTTTTTT", "MKLLVVVFCLVLAAP"]'
    start_time=$(date +%s.%N)
    curl -s -X POST "$API_URL/api/v1/predict/batch" \
        -H "Content-Type: application/json" \
        -d "{\"sequences\": $test_sequences}" > /dev/null
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    
    if (( $(echo "$duration < 30.0" | bc -l) )); then
        print_success "批次預測響應時間: ${duration}秒 (< 30秒)"
    else
        print_error "批次預測響應時間過長: ${duration}秒 (期望 < 30秒)"
    fi
}

# 測試 8: 容器日誌檢查
test_container_logs() {
    print_header "8. 容器日誌檢查"
    
    # 檢查是否有錯誤日誌
    error_count=$(docker logs "$CONTAINER_NAME" 2>&1 | grep -i "error" | wc -l | tr -d ' ')
    
    if [ "$error_count" -eq 0 ]; then
        print_success "未發現錯誤日誌"
    else
        print_error "發現 $error_count 個錯誤日誌"
        print_info "最近的錯誤日誌:"
        docker logs "$CONTAINER_NAME" 2>&1 | grep -i "error" | tail -5
    fi
    
    # 檢查模型加載日誌
    if docker logs "$CONTAINER_NAME" 2>&1 | grep -qi "model.*load"; then
        print_success "發現模型加載日誌"
    else
        print_info "未發現模型加載日誌（可能是延遲加載）"
    fi
}

# 主函數
main() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════╗"
    echo "║  Multi-AOP FastAPI Docker 測試腳本   ║"
    echo "╚════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # 等待服務就緒
    if ! wait_for_service; then
        print_error "服務未就緒，請檢查容器狀態"
        exit 1
    fi
    
    # 執行測試
    test_environment || exit 1
    test_health_check
    test_model_info
    test_single_prediction
    test_batch_prediction
    test_error_scenarios
    test_performance
    test_container_logs
    
    # 測試總結
    print_header "測試總結"
    echo -e "總測試數: ${BLUE}$TESTS_TOTAL${NC}"
    echo -e "通過: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "失敗: ${RED}$TESTS_FAILED${NC}"
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "\n${GREEN}✓ 所有測試通過！${NC}\n"
        exit 0
    else
        echo -e "\n${RED}✗ 部分測試失敗${NC}\n"
        exit 1
    fi
}

# 執行主函數
main "$@"

