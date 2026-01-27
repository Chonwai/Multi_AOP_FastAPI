# -*- coding: utf-8 -*-
"""
適配器異常處理

定義適配器層特定的異常類型，便於錯誤追蹤和處理
"""


class AdapterError(Exception):
    """適配器基礎異常"""
    pass


class CoreImportError(AdapterError):
    """核心模塊導入失敗"""
    def __init__(self, module_name: str, original_error: Exception):
        self.module_name = module_name
        self.original_error = original_error
        super().__init__(
            f"Failed to import core module '{module_name}': {str(original_error)}"
        )


class CoreVersionMismatchError(AdapterError):
    """核心算法版本不匹配"""
    def __init__(self, expected_version: str, actual_version: str):
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Core algorithm version mismatch. Expected: {expected_version}, Actual: {actual_version}"
        )


class ModelAdaptationError(AdapterError):
    """模型適配失敗"""
    def __init__(self, message: str, model_path: str = None):
        self.model_path = model_path
        super().__init__(message)


class DataAdaptationError(AdapterError):
    """數據適配失敗"""
    def __init__(self, message: str, sequence: str = None):
        self.sequence = sequence
        super().__init__(message)
