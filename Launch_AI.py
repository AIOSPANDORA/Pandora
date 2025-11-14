"""
Universal AIOS Launcher for NVIDIA/Intel/ASUS ROG

- Detects platform & hardware.
- Verifies NVIDIA GPU (CUDA) and Intel CPU presence.
- Launches main AI/Orchestrator routine.
"""

import platform
import os
import subprocess
import sys

def has_nvidia_gpu():
    try:
        # Works on Windows and most Linux distros
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except Exception:
        return False

def has_intel_cpu():
    try:
        if platform.system() == "Windows":
            output = subprocess.check_output("wmic cpu get name", shell=True)
            return b'Intel' in output
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                return "Intel" in f.read()
    except Exception:
        return False

def asus_rog_present():
    # Basic check for ASUS ROG in system product or vendor strings
    try:
        if platform.system() == "Windows":
            output = subprocess.check_output('wmic computersystem get manufacturer,model', shell=True)
            return b'ASUSTeK' in output or b'ROG' in output
        elif platform.system() == "Linux":
            with open("/sys/class/dmi/id/board_vendor") as f:
                if "ASUS" in f.read() or "ROG" in f.read():
                    return True
    except Exception:
        return False

def main():
    print("\n==== AIOS/Pandora Universal Hardware Launcher ====\n")
    print(f"OS: {platform.system()}")
    print("-> Checking for NVIDIA GPU:", "✓" if has_nvidia_gpu() else "✗")
    print("-> Checking for Intel CPU: ", "✓" if has_intel_cpu() else "✗")
    print("-> Checking for ASUS ROG:  ", "✓" if asus_rog_present() else "✗")
    # Insert any additional OS/hardware-specific setup or guidance here

    # LAUNCH AI/Pandora or instruct user to install needed drivers
    if not (has_intel_cpu() and has_nvidia_gpu()):
        print("WARNING: Intel CPU and/or NVIDIA GPU not detected! Exiting.")
        sys.exit(1)

    print("System compatible. Launching core orchestrator.")
    # Import and run main orchestrator here (replace example below!):
    # from pandora_fabric_orchestrator import PandoraFabricOrchestrator
    # PandoraFabricOrchestrator().start()

if __name__ == "__main__":
    main()