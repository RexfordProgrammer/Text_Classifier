#!/usr/bin/env python3
import sys
import json
from pathlib import Path

def main():
    local_json_path = "local.json"
    file_path = Path(local_json_path)
    
    if not file_path.exists():
        print(f"[error] File not found: {file_path}")
        sys.exit(1)

    try:
        text = file_path.read_text(encoding="utf-8")
        json.loads(text)
        print(f"[ok] {file_path} contains valid JSON")
    except json.JSONDecodeError as e:
        print(f"[fail] Invalid JSON in {file_path}: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
