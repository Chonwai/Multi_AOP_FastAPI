#!/usr/bin/env python3
"""
純 Python 運行腳本（完全不使用 uvicorn）
使用 Python 內建的 HTTP 服務器（僅用於 Demo/MVP 測試）

注意：這個方法性能較低，不適合生產環境
僅用於快速測試和調試

使用方法:
    python run_pure_python.py
"""

import os
import sys
import asyncio
from pathlib import Path
from wsgiref.simple_server import make_server

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.main import app
from app.config import settings
from app.utils.logging_config import setup_logging, get_logger

# 設置日誌
setup_logging()
logger = get_logger(__name__)

# 將 FastAPI ASGI 應用轉換為 WSGI（使用 asgiref）
try:
    from asgiref.wsgi import WsgiToAsgi
    
    # 將 ASGI 應用轉換為 WSGI
    wsgi_app = WsgiToAsgi(app)
    
    logger.info("=" * 60)
    logger.info("啟動 Multi-AOP FastAPI 應用程式（純 Python 模式）")
    logger.info("注意：此模式性能較低，僅用於測試")
    logger.info("=" * 60)
    
    host = os.getenv("API_HOST", settings.API_HOST)
    port = int(os.getenv("API_PORT", settings.API_PORT))
    
    logger.info(f"服務運行在: http://{host}:{port}")
    logger.info(f"訪問文檔: http://localhost:{port}/docs")
    logger.info(f"健康檢查: http://localhost:{port}/health")
    logger.info("按 Ctrl+C 停止服務")
    logger.info("=" * 60)
    
    # 使用 Python 內建的 WSGI 服務器
    with make_server(host, port, wsgi_app) as httpd:
        logger.info(f"服務器已啟動在 {host}:{port}")
        httpd.serve_forever()
        
except ImportError:
    logger.error("需要安裝 asgiref: pip install asgiref")
    logger.info("或者使用 run_simple.py（推薦）")
    sys.exit(1)
except KeyboardInterrupt:
    logger.info("應用程式已停止")
except Exception as e:
    logger.error(f"啟動失敗: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

