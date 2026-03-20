#!/bin/bash
# sync_unity.sh — Copy C# bridge scripts to the Unity Assets folder.
# Run this after editing any .cs file in unity_bridge/.

REPO_DIR="$(cd "$(dirname "$0")" && pwd)/unity_bridge"
UNITY_DIR="/Users/ayush/cherry/Assets"

for f in CherryUnityBridge.cs CherryCamera.cs; do
    src="$REPO_DIR/$f"
    dst="$UNITY_DIR/$f"
    if ! diff -q "$src" "$dst" > /dev/null 2>&1; then
        cp "$src" "$dst"
        echo "Synced: $f"
    else
        echo "Up to date: $f"
    fi
done
