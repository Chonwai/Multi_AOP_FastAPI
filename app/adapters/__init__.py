# -*- coding: utf-8 -*-
"""
Adapters Package - 適配器層

這個包負責封裝對上游核心算法(predict/)的調用。
當 Algorithm Engineer 更新核心算法時，只需修改這個包中的適配器。

職責：
1. 導入上游核心算法模塊
2. 提供統一的接口給 FastAPI 服務層
3. 隔離核心算法變化對應用層的影響

設計模式：
- Adapter Pattern: 適配不同版本的核心算法
- Facade Pattern: 簡化複雜的核心算法調用
- Dependency Injection: 便於單元測試
"""

from app.adapters.core_adapter import CoreAdapter, get_core_adapter

__all__ = [
    'CoreAdapter',
    'get_core_adapter',
]
