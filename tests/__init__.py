"""
Quantum Astra Test Suite Package
"""

import sys
import os
import importlib.util

# Ensure src is on sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

src_dir = os.path.join(base_dir, "src")
if "src" not in sys.modules:
    spec = importlib.util.spec_from_file_location("src", os.path.join(src_dir, "__init__.py"))
    src_mod = importlib.util.module_from_spec(spec)
    sys.modules["src"] = src_mod
    spec.loader.exec_module(src_mod)
