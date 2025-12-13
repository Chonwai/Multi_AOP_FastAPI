#!/bin/bash
# 本地部署兼容性測試腳本
# 測試 Dockerfile PORT 修改是否影響本地部署

set -e  # 遇到錯誤立即退出

echo "================================================"
echo "  本地部署兼容性測試"
echo "  測試 Dockerfile PORT 修改的向後兼容性"
echo "================================================"
echo ""

# 顏色定義
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 測試結果統計
PASSED=0
FAILED=0

# 測試函數
test_endpoint() {
    local url=$1
    local test_name=$2
    
    echo -n "  測試 $test_name ... "
    
    if curl -s -f "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ 通過${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗ 失敗${NC}"
        ((FAILED++))
        return 1
    fi
}

# 清理函數
cleanup() {
    echo ""
    echo "清理測試環境..."
    docker-compose -f docker/docker-compose.yml down -v > /dev/null 2>&1 || true
    docker rm -f test-container > /dev/null 2>&1 || true
    rm -f .env
    unset PORT
    echo "清理完成"
}

# 設置清理陷阱
trap cleanup EXIT

echo "📋 測試計劃:"
echo "  1. 默認配置測試（PORT 未設置）"
echo "  2. 自定義端口測試（PORT=9000）"
echo "  3. .env 文件測試"
echo "  4. 直接 Docker 運行測試"
echo ""

# ============================================
# 測試 1: 默認配置（PORT 未設置）
# ============================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "測試 1: 默認配置（PORT 未設置）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 清理環境
unset PORT
rm -f .env

echo "啟動服務（默認端口 8000）..."
docker-compose -f docker/docker-compose.yml up -d > /dev/null 2>&1

echo "等待服務啟動（30秒）..."
sleep 30

echo "執行測試:"
test_endpoint "http://localhost:8000/health" "健康檢查"
test_endpoint "http://localhost:8000/" "根路徑"
test_endpoint "http://localhost:8000/docs" "API 文檔"

echo "停止服務..."
docker-compose -f docker/docker-compose.yml down > /dev/null 2>&1

echo -e "${GREEN}✓ 測試 1 完成${NC}"
echo ""

# ============================================
# 測試 2: 自定義端口（PORT=9000）
# ============================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "測試 2: 自定義端口（PORT=9000）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

export PORT=9000

echo "啟動服務（自定義端口 9000）..."
docker-compose -f docker/docker-compose.yml up -d > /dev/null 2>&1

echo "等待服務啟動（30秒）..."
sleep 30

echo "執行測試:"
test_endpoint "http://localhost:9000/health" "健康檢查"
test_endpoint "http://localhost:9000/" "根路徑"
test_endpoint "http://localhost:9000/docs" "API 文檔"

echo "停止服務..."
docker-compose -f docker/docker-compose.yml down > /dev/null 2>&1

unset PORT

echo -e "${GREEN}✓ 測試 2 完成${NC}"
echo ""

# ============================================
# 測試 3: .env 文件
# ============================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "測試 3: .env 文件配置"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cat > .env << EOF
PORT=7000
ENVIRONMENT=development
LOG_LEVEL=DEBUG
EOF

echo "創建 .env 文件（PORT=7000）"

echo "啟動服務..."
docker-compose -f docker/docker-compose.yml up -d > /dev/null 2>&1

echo "等待服務啟動（30秒）..."
sleep 30

echo "執行測試:"
test_endpoint "http://localhost:7000/health" "健康檢查"
test_endpoint "http://localhost:7000/" "根路徑"
test_endpoint "http://localhost:7000/docs" "API 文檔"

echo "停止服務..."
docker-compose -f docker/docker-compose.yml down > /dev/null 2>&1

rm -f .env

echo -e "${GREEN}✓ 測試 3 完成${NC}"
echo ""

# ============================================
# 測試 4: 直接 Docker 運行
# ============================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "測試 4: 直接 Docker 運行"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "構建 Docker image..."
docker build -f docker/Dockerfile -t multi-aop-test . > /dev/null 2>&1

echo "運行容器（默認端口 8000）..."
docker run -d --name test-container -p 8000:8000 multi-aop-test > /dev/null 2>&1

echo "等待服務啟動（30秒）..."
sleep 30

echo "執行測試:"
test_endpoint "http://localhost:8000/health" "健康檢查"
test_endpoint "http://localhost:8000/" "根路徑"

echo "停止容器..."
docker stop test-container > /dev/null 2>&1
docker rm test-container > /dev/null 2>&1

echo "運行容器（自定義端口 9000）..."
docker run -d --name test-container -p 9000:9000 -e PORT=9000 multi-aop-test > /dev/null 2>&1

echo "等待服務啟動（30秒）..."
sleep 30

echo "執行測試:"
test_endpoint "http://localhost:9000/health" "健康檢查"
test_endpoint "http://localhost:9000/" "根路徑"

echo "停止容器..."
docker stop test-container > /dev/null 2>&1
docker rm test-container > /dev/null 2>&1

echo "清理 Docker image..."
docker rmi multi-aop-test > /dev/null 2>&1

echo -e "${GREEN}✓ 測試 4 完成${NC}"
echo ""

# ============================================
# 測試結果總結
# ============================================
echo "================================================"
echo "  測試結果總結"
echo "================================================"
echo ""
echo "總測試數: $((PASSED + FAILED))"
echo -e "${GREEN}通過: $PASSED${NC}"
echo -e "${RED}失敗: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  ✓ 所有測試通過！${NC}"
    echo -e "${GREEN}  ✓ Dockerfile 修改完全向後兼容！${NC}"
    echo -e "${GREEN}  ✓ 本地部署不受影響！${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 0
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}  ✗ 部分測試失敗${NC}"
    echo -e "${RED}  請檢查錯誤日誌${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 1
fi

