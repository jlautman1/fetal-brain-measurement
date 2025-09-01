#!/usr/bin/env python3
"""
Validation script for OpenRecon Fetal Brain Measurement Integration
"""

import os
import sys

def main():
    print("=" * 60)
    print("OpenRecon Fetal Brain Measurement Integration Validator")
    print("=" * 60)
    print()
    
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    print()
    
    # Check core files
    print("1. Checking core integration files...")
    
    files_to_check = [
        "Dockerfile.openrecon.integrated",
        "openrecon.py",              # Main OpenRecon i2i module (proper structure)
        "openrecon.json",            # Configuration file
        "build-openrecon-image.bat",
        "run-openrecon-server.bat", 
        "test-client.py",
        "deployToOpenRecon.md"       # Updated deployment guide
    ]
    
    all_found = True
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            all_found = False
    
    print()
    print("2. Checking fetal brain components...")
    
    fetal_files = [
        "Code/FetalMeasurements-master/execute.py",
        "Code/FetalMeasurements-master/requirements.txt",
        "Models/"
    ]
    
    for file in fetal_files:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            all_found = False
    
    print()
    if all_found:
        print("✅ ALL FILES FOUND!")
        print("You can proceed with building the Docker image.")
    else:
        print("❌ Some files are missing. Please check the setup.")
    
    return all_found

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)