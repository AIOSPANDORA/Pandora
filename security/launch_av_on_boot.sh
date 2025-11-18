#!/bin/bash
# This script ensures the Pandora antivirus daemon starts on system boot.

# 1. Enable ClamAV daemon on boot (Debian/Ubuntu):
sudo systemctl enable clamav-daemon

# 2. Start ClamAV daemon right now:
sudo systemctl start clamav-daemon

# 3. Optional: Schedule the antivirus launcher with systemd at boot

# Create a systemd service unit for the Pandora Antivirus Launcher
SERVICE_FILE="/etc/systemd/system/pandora-antivirus-launcher.service"

sudo bash -c "cat > $SERVICE_FILE" <<EOL
[Unit]
Description=Pandora Antivirus Launcher
After=network.target clamav-daemon.service

[Service]
Type=simple
ExecStart=/usr/bin/python3 /path/to/security/antivirus_launcher.py
Restart=always
User=root

[Install]
WantedBy=multi-user.target
EOL

# Reload the systemd manager configuration
sudo systemctl daemon-reload

# Enable the Pandora Antivirus Launcher service to autostart
sudo systemctl enable pandora-antivirus-launcher

# Start it immediately (optional)
sudo systemctl start pandora-antivirus-launcher

echo "Pandora Antivirus is now set to auto-launch on system boot!"
echo "Make sure to edit ExecStart=/usr/bin/python3 /path/to/security/antivirus_launcher.py with the correct path."