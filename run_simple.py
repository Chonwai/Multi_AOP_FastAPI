#!/usr/bin/env python3
"""
簡單的 Python 運行腳本（不使用 uvicorn）
適用於 Demo/MVP 階段的快速測試

使用方法:
    python run_simple.py

或者設置環境變量:
    API_HOST=0.0.0.0 API_PORT=8000 python run_simple.py
"""

import os
import sys
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.main import app
from app.config import settings
from app.utils.logging_config import setup_logging, get_logger

# 設置日誌
setup_logging()
logger = get_logger(__name__)

if __name__ == "__main__":
    import uvicorn
    
    # 從環境變量或配置獲取主機和端口
    host = os.getenv("API_HOST", settings.API_HOST)
    port = int(os.getenv("API_PORT", settings.API_PORT))
    
    logger.info("=" * 60)
    logger.info("啟動 Multi-AOP FastAPI 應用程式（簡單模式）")
    logger.info(f"主機: {host}")
    logger.info(f"端口: {port}")
    logger.info(f"環境: {settings.ENVIRONMENT}")
    logger.info(f"模型路徑: {settings.MODEL_PATH}")
    logger.info(f"設備: {settings.DEVICE}")
    logger.info("=" * 60)
    logger.info("訪問文檔: http://localhost:{}/docs".format(port))
    logger.info("健康檢查: http://localhost:{}/health".format(port))
    logger.info("=" * 60)
    
    # 使用 uvicorn 運行（但這是通過 Python 腳本調用，更容易調試）
    # 對於純 Python 運行，可以使用 hypercorn 或其他 ASGI 服務器
    # 但 uvicorn 是最簡單和 FastAPI 官方推薦的方式
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=settings.LOG_LEVEL.lower(),
            # 開發模式：自動重載
            reload=settings.ENVIRONMENT == "development",
            # 訪問日誌
            access_log=True,
        )
    except KeyboardInterrupt:
        logger.info("應用程式已停止")
    except Exception as e:
        logger.error(f"啟動失敗: {str(e)}")
        sys.exit(1)

