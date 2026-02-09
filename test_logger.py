#!/usr/bin/env python
"""Quick test to verify logger is working correctly"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

# Test 1: Logger with default directory
print("Test 1: Logger with default directory")
logger1 = get_logger(log_dir="test_logs_default")
logger1.info("This is test message 1")
logger1.debug("This is debug message 1")
logger1.warning("This is warning message 1")
log_file1 = logger1.get_log_file()
print(f"  Log file: {log_file1}")

# Test 2: Logger with custom directory
print("\nTest 2: Logger with custom directory")
logger2 = get_logger(log_dir="test_logs_custom")
logger2.info("This is test message 2")
logger2.error("This is error message 2")
log_file2 = logger2.get_log_file()
print(f"  Log file: {log_file2}")

# Test 3: Verify file content
print("\nTest 3: Verifying log files")
if Path(log_file1).exists():
    with open(log_file1, 'r') as f:
        content1 = f.read()
    print(f"✓ Log file 1 exists with {len(content1)} bytes")
    if len(content1) > 0:
        print("  First 200 chars:")
        print("  " + content1[:200].replace('\n', '\n  '))
    else:
        print("  WARNING: Log file is empty!")
else:
    print(f"✗ Log file 1 not found: {log_file1}")

if Path(log_file2).exists():
    with open(log_file2, 'r') as f:
        content2 = f.read()
    print(f"✓ Log file 2 exists with {len(content2)} bytes")
    if len(content2) > 0:
        print("  First 200 chars:")
        print("  " + content2[:200].replace('\n', '\n  '))
    else:
        print("  WARNING: Log file is empty!")
else:
    print(f"✗ Log file 2 not found: {log_file2}")

print("\n✓ Logger test complete!")
