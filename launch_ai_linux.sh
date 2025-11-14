#!/bin/bash
echo "== AIOS/Pandora Universal Launcher =="
if ! command -v python3 &> /dev/null
then
    echo "Python3 not found! Please install python3."
    exit 1
fi
python3 Launch_AI.py