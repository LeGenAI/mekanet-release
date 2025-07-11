#!/usr/bin/env python3
import sys
print("Testing basic imports...")

try:
    import numpy as np
    print("✓ numpy")
except ImportError as e:
    print(f"✗ numpy: {e}")

try:
    import pandas as pd
    print("✓ pandas")
except ImportError as e:
    print(f"✗ pandas: {e}")

try:
    import yaml
    print("✓ yaml")
except ImportError as e:
    print(f"✗ yaml: {e}")

try:
    config_path = 'experiments/detection/configs/paper_reproduction_quick.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Config file loaded: {len(config)} experiments")
except Exception as e:
    print(f"✗ Config loading failed: {e}")

print("Basic test completed!")
