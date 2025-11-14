#!/bin/bash
echo "Installing AIOS/Pandora requirements"
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-psutil
pip3 install --upgrade pip
pip3 install psutil
# NVIDIA & Intel drivers (if needed)
sudo apt-get install -y nvidia-driver-535
echo "Done."