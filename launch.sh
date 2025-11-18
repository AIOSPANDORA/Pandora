#!/bin/bash
# Pandora Universal Startup - Cubic/USB Optimized

cd "$(dirname "$0")"
# Start antivirus firewall (background)
bash security/launch_av_on_boot.sh &
# Start adaptive firewall (background, optional)
python3 security/fluid_firewall.py &

# Main AIOS startup sequence
cd startup
python3 health_monitor.py --boot || {
    echo "System health fault! Launching safe mode."
    python3 safe_mode.py
    exit 1
}
python3 pandora_launcher.py || {
    echo "Orchestrator error! Safe mode fallback."
    python3 safe_mode.py
}