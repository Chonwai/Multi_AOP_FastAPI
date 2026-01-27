#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Architecture Verification Test

Verifies that the Adapter Pattern is correctly implemented
WITHOUT requiring all dependencies to be installed.
"""

import sys
import os
import ast
import importlib.util

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("Architecture Verification: Adapter Pattern Implementation")
print("=" * 80)

def check_imports_in_file(filepath, forbidden_patterns):
    """Check if a file contains forbidden import patterns"""
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for pattern in forbidden_patterns:
                        if module.startswith(pattern):
                            return False, f"Found forbidden import: from {module}"
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        for pattern in forbidden_patterns:
                            if alias.name.startswith(pattern):
                                return False, f"Found forbidden import: import {alias.name}"
            return True, "OK"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

print("\n[Test 1] Verify Service Layer Isolation")
print("Services should NOT import from predict/ or reproduce/ directly")

service_files = [
    "app/services/model_manager.py",
    "app/services/predictor.py"
]

forbidden_imports = ["predict.", "reproduce."]
all_passed = True

for service_file in service_files:
    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), service_file)
    if os.path.exists(filepath):
        passed, msg = check_imports_in_file(filepath, forbidden_imports)
        status = "✓" if passed else "✗"
        print(f"  {status} {service_file}: {msg}")
        if not passed:
            all_passed = False
    else:
        print(f"  ⚠ {service_file}: File not found")

print("\n[Test 2] Verify Adapter Implementation")
print("Adapter SHOULD import from predict/ directory (via sys.path)")

adapter_file = "app/adapters/core_adapter.py"
filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), adapter_file)

if os.path.exists(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        # Check for either direct import or sys.path manipulation + import
        has_predict_setup = 'PREDICT_DIR' in content and 'sys.path' in content
        has_core_imports = 'from aop_def import' in content or 'from predict.aop_def import' in content
        
        if has_predict_setup and has_core_imports:
            print(f"  ✓ {adapter_file}: Correctly imports core algorithms")
        else:
            print(f"  ✗ {adapter_file}: Missing core algorithm imports")
            if not has_predict_setup:
                print(f"    - Missing predict/ directory setup")
            if not has_core_imports:
                print(f"    - Missing core algorithm imports")
            all_passed = False
else:
    print(f"  ✗ {adapter_file}: File not found")
    all_passed = False

print("\n[Test 3] Verify processors.py delegates to Adapter")
processors_file = "app/core/data/processors.py"
filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), processors_file)

if os.path.exists(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        has_adapter_import = 'from app.adapters.core_adapter import get_core_adapter' in content
        has_adapter_calls = '_adapter.' in content
        
        if has_adapter_import and has_adapter_calls:
            print(f"  ✓ {processors_file}: Correctly delegates to CoreAdapter")
        else:
            print(f"  ✗ {processors_file}: Missing Adapter delegation")
            all_passed = False
else:
    print(f"  ✗ {processors_file}: File not found")
    all_passed = False

print("\n[Test 4] Verify ModelManager uses Adapter")
model_manager_file = "app/services/model_manager.py"
filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_manager_file)

if os.path.exists(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        has_adapter_import = 'from app.adapters.core_adapter import get_core_adapter' in content
        has_adapter_usage = 'adapter.load_model' in content
        
        if has_adapter_import and has_adapter_usage:
            print(f"  ✓ {model_manager_file}: Uses CoreAdapter for model loading")
        else:
            print(f"  ✗ {model_manager_file}: Not using CoreAdapter properly")
            all_passed = False
else:
    print(f"  ✗ {model_manager_file}: File not found")
    all_passed = False

print("\n[Test 5] Check Adapter Interface Definition")
interface_file = "app/adapters/interfaces.py"
filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), interface_file)

if os.path.exists(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        has_predictor_core = 'class IPredictorCore' in content
        has_data_processor = 'class IDataProcessor' in content
        has_model_info = 'class IModelInfo' in content
        
        if has_predictor_core and has_data_processor and has_model_info:
            print(f"  ✓ {interface_file}: All interfaces defined")
        else:
            missing = []
            if not has_predictor_core: missing.append("IPredictorCore")
            if not has_data_processor: missing.append("IDataProcessor")
            if not has_model_info: missing.append("IModelInfo")
            print(f"  ✗ {interface_file}: Missing interfaces: {', '.join(missing)}")
            all_passed = False
else:
    print(f"  ✗ {interface_file}: File not found")
    all_passed = False

print("\n" + "=" * 80)
print("Architecture Verification Summary")
print("=" * 80)

if all_passed:
    print("✓ ALL TESTS PASSED!")
    print("\nArchitecture is correctly implemented:")
    print("  1. Service layer is isolated from predict/ directory")
    print("  2. CoreAdapter properly wraps predict/ imports")
    print("  3. processors.py delegates to Adapter")
    print("  4. ModelManager uses Adapter for model operations")
    print("  5. All required interfaces are defined")
    print("\nDependency flow verified:")
    print("  app/services/ → app/adapters/ → predict/")
    print("  ✓ No direct dependencies from services to core algorithms")
else:
    print("✗ SOME TESTS FAILED")
    print("Please review the failed tests above")
    sys.exit(1)

print("=" * 80)
