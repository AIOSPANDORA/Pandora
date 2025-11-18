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
    """Check for NVIDIA GPU with nvidia-smi"""
    try:
        # Works on Windows and most Linux distros
        result = subprocess.run(['nvidia-smi'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              timeout=5)
        return result.returncode == 0
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Warning: Could not check for NVIDIA GPU: {e}")
        return False

def has_intel_cpu():
    """Check for Intel CPU"""
    try:
        if platform.system() == "Windows":
            output = subprocess.check_output("wmic cpu get name", 
                                           shell=True,
                                           timeout=5)
            return b'Intel' in output
        elif platform.system() == "Linux":
            if os.path.exists("/proc/cpuinfo"):
                with open("/proc/cpuinfo", "r") as f:
                    return "Intel" in f.read()
        return False
    except Exception as e:
        print(f"Warning: Could not detect CPU type: {e}")
        return False

def asus_rog_present():
    """Basic check for ASUS ROG in system product or vendor strings"""
    try:
        if platform.system() == "Windows":
            output = subprocess.check_output('wmic computersystem get manufacturer,model',
                                           shell=True,
                                           timeout=5)
            return b'ASUSTeK' in output or b'ROG' in output
        elif platform.system() == "Linux":
            # Check DMI information
            dmi_paths = [
                "/sys/class/dmi/id/board_vendor",
                "/sys/class/dmi/id/product_name"
            ]
            for path in dmi_paths:
                if os.path.exists(path):
                    with open(path) as f:
                        content = f.read()
                        if "ASUS" in content or "ROG" in content:
                            return True
        return False
    except Exception as e:
        print(f"Info: Could not detect ASUS ROG: {e}")
        return False

def check_python_version():
    """Verify Python version is adequate"""
    version = sys.version_info
    if version < (3, 8):
        print(f"WARNING: Python {version.major}.{version.minor} detected.")
        print("Python 3.8 or higher is recommended for optimal compatibility.")
        return False
    return True

def check_dependencies():
    """Check if critical dependencies are available"""
    missing = []
    
    try:
        import psutil
    except ImportError:
        missing.append("psutil")
    
    if missing:
        print(f"WARNING: Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    return True

def main():
    """Main launcher with comprehensive hardware detection"""
    print("\n==== AIOS/Pandora Universal Hardware Launcher ====\n")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Hardware checks
    has_gpu = has_nvidia_gpu()
    has_cpu = has_intel_cpu()
    has_rog = asus_rog_present()
    
    print("-> Checking for NVIDIA GPU:", "✓" if has_gpu else "✗")
    print("-> Checking for Intel CPU: ", "✓" if has_cpu else "✗")
    print("-> Checking for ASUS ROG:  ", "✓" if has_rog else "✗")
    
    # Software checks
    python_ok = check_python_version()
    deps_ok = check_dependencies()
    
    print("\n==== Compatibility Assessment ====")
    
    # Determine launch strategy
    warnings = []
    can_launch = True
    
    if not has_gpu:
        warnings.append("NVIDIA GPU not detected - GPU-accelerated features unavailable")
    
    if not has_cpu:
        warnings.append("Intel CPU not detected - some optimizations may be unavailable")
    
    if not python_ok:
        warnings.append("Python version is below recommended (3.8+)")
    
    if not deps_ok:
        warnings.append("Missing required dependencies")
        can_launch = False
    
    if warnings:
        print("\n⚠ Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not can_launch:
        print("\n✗ Cannot launch: Please install missing dependencies first.")
        print("  Run: pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n✓ System compatible. Ready to launch core orchestrator.")
    print("\nTo start Pandora AIOS:")
    print("  - GUI mode: python3 pandora_gui.py")
    print("  - Web portal: python3 assimilate.py")
    print("  - Monitor: python3 system_monitoring.py")
    
    # Future: Import and run main orchestrator here
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