"""
Pandora AIOS Diagnostic System
-------------------------------
Performs comprehensive diagnostics while running scripts and during AI thinking.
Reports system health, resource usage, dependencies, and potential issues.

Philosophy: Transparency, self-examination, continuous monitoring
"""

import os
import sys
import platform
import subprocess
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

try:
    import psutil
except ImportError:
    psutil = None

class PandoraDiagnostics:
    """Comprehensive diagnostic system for Pandora AIOS"""
    
    def __init__(self, log_path: str = "/var/log/pandora_diagnostics.log"):
        self.log_path = log_path
        self.start_time = time.time()
        self.diagnostics_history = []
        self._ensure_log_directory()
    
    def _ensure_log_directory(self):
        """Ensure log directory exists"""
        log_dir = os.path.dirname(self.log_path)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except PermissionError:
                # Fallback to temp directory
                self.log_path = f"/tmp/pandora_diagnostics_{os.getpid()}.log"
    
    def log(self, level: str, message: str):
        """Log diagnostic message"""
        timestamp = datetime.utcnow().isoformat()
        entry = f"[{timestamp}][{level}] {message}"
        
        try:
            with open(self.log_path, "a") as f:
                f.write(entry + "\n")
        except Exception as e:
            print(f"[WARNING] Could not write to log: {e}")
        
        print(entry)
    
    def check_system_info(self) -> Dict[str, Any]:
        """Collect basic system information"""
        self.log("INFO", "Collecting system information...")
        
        info = {
            "timestamp": datetime.utcnow().isoformat(),
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "hostname": platform.node(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
        }
        
        self.log("INFO", f"Platform: {info['platform']} {info['platform_release']}")
        self.log("INFO", f"Architecture: {info['architecture']}")
        self.log("INFO", f"Python: {info['python_version']}")
        
        return info
    
    def check_hardware(self) -> Dict[str, Any]:
        """Check hardware resources and availability"""
        self.log("INFO", "Checking hardware resources...")
        
        hardware = {
            "cpu_count": None,
            "cpu_percent": None,
            "memory_total": None,
            "memory_available": None,
            "memory_percent": None,
            "disk_total": None,
            "disk_free": None,
            "disk_percent": None,
        }
        
        if psutil:
            hardware["cpu_count"] = psutil.cpu_count(logical=True)
            hardware["cpu_percent"] = psutil.cpu_percent(interval=1)
            
            mem = psutil.virtual_memory()
            hardware["memory_total"] = mem.total
            hardware["memory_available"] = mem.available
            hardware["memory_percent"] = mem.percent
            
            disk = psutil.disk_usage('/')
            hardware["disk_total"] = disk.total
            hardware["disk_free"] = disk.free
            hardware["disk_percent"] = disk.percent
            
            self.log("INFO", f"CPU: {hardware['cpu_count']} cores, {hardware['cpu_percent']}% used")
            self.log("INFO", f"Memory: {hardware['memory_available'] / (1024**3):.2f}GB free / {hardware['memory_total'] / (1024**3):.2f}GB total ({hardware['memory_percent']}% used)")
            self.log("INFO", f"Disk: {hardware['disk_free'] / (1024**3):.2f}GB free / {hardware['disk_total'] / (1024**3):.2f}GB total ({hardware['disk_percent']}% used)")
        else:
            self.log("WARNING", "psutil not available - limited hardware diagnostics")
        
        return hardware
    
    def check_nvidia_gpu(self) -> Dict[str, Any]:
        """Check NVIDIA GPU availability and status"""
        self.log("INFO", "Checking NVIDIA GPU...")
        
        gpu_info = {
            "available": False,
            "driver_version": None,
            "cuda_version": None,
            "gpus": []
        }
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu',
                 '--format=csv,noheader'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            
            if result.returncode == 0:
                gpu_info["available"] = True
                output = result.stdout.decode('utf-8').strip()
                
                for line in output.split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 8:
                            gpu_info["gpus"].append({
                                "index": parts[0],
                                "name": parts[1],
                                "driver_version": parts[2],
                                "memory_total": parts[3],
                                "memory_free": parts[4],
                                "memory_used": parts[5],
                                "temperature": parts[6],
                                "utilization": parts[7]
                            })
                
                self.log("INFO", f"NVIDIA GPU(s) detected: {len(gpu_info['gpus'])} device(s)")
                for gpu in gpu_info["gpus"]:
                    self.log("INFO", f"  GPU {gpu['index']}: {gpu['name']} (Temp: {gpu['temperature']}, Util: {gpu['utilization']})")
            else:
                self.log("WARNING", "nvidia-smi command failed - GPU not available or driver issue")
                
        except FileNotFoundError:
            self.log("WARNING", "nvidia-smi not found - NVIDIA drivers not installed")
        except subprocess.TimeoutExpired:
            self.log("ERROR", "nvidia-smi timeout - GPU may be unresponsive")
        except Exception as e:
            self.log("ERROR", f"GPU check error: {e}")
        
        return gpu_info
    
    def check_intel_cpu(self) -> Dict[str, Any]:
        """Check if Intel CPU is present"""
        self.log("INFO", "Checking for Intel CPU...")
        
        cpu_info = {
            "is_intel": False,
            "cpu_name": None,
            "details": None
        }
        
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    if "Intel" in cpuinfo:
                        cpu_info["is_intel"] = True
                        # Extract model name
                        for line in cpuinfo.split('\n'):
                            if 'model name' in line:
                                cpu_info["cpu_name"] = line.split(':')[1].strip()
                                break
                        cpu_info["details"] = "Intel CPU detected from /proc/cpuinfo"
                        
            elif platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5
                )
                output = result.stdout.decode('utf-8')
                if 'Intel' in output:
                    cpu_info["is_intel"] = True
                    lines = output.strip().split('\n')
                    if len(lines) > 1:
                        cpu_info["cpu_name"] = lines[1].strip()
                    cpu_info["details"] = "Intel CPU detected from wmic"
            
            if cpu_info["is_intel"]:
                self.log("INFO", f"Intel CPU detected: {cpu_info['cpu_name']}")
            else:
                self.log("WARNING", "Intel CPU not detected")
                
        except Exception as e:
            self.log("ERROR", f"CPU check error: {e}")
        
        return cpu_info
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check Python dependencies and system packages"""
        self.log("INFO", "Checking dependencies...")
        
        deps = {
            "python_packages": {},
            "system_packages": {},
            "missing": []
        }
        
        # Check Python packages
        required_python = [
            "psutil",
            "flask",
            "flask_qrcode",
            "pyclamd",
            "inotify_simple"
        ]
        
        for pkg in required_python:
            try:
                __import__(pkg)
                deps["python_packages"][pkg] = "installed"
                self.log("INFO", f"Python package '{pkg}': OK")
            except ImportError:
                deps["python_packages"][pkg] = "missing"
                deps["missing"].append(f"python:{pkg}")
                self.log("WARNING", f"Python package '{pkg}': NOT FOUND")
        
        # Check system packages (Linux)
        if platform.system() == "Linux":
            system_pkgs = ["ufw", "clamav", "clamav-daemon"]
            for pkg in system_pkgs:
                result = subprocess.run(
                    ["which", pkg.replace("-daemon", "d")],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                if result.returncode == 0:
                    deps["system_packages"][pkg] = "installed"
                    self.log("INFO", f"System package '{pkg}': OK")
                else:
                    deps["system_packages"][pkg] = "missing"
                    deps["missing"].append(f"system:{pkg}")
                    self.log("WARNING", f"System package '{pkg}': NOT FOUND")
        
        return deps
    
    def check_security_services(self) -> Dict[str, Any]:
        """Check if security services are running"""
        self.log("INFO", "Checking security services...")
        
        services = {
            "clamav": False,
            "ufw": False,
            "firewall_active": False
        }
        
        if platform.system() == "Linux":
            # Check ClamAV
            try:
                result = subprocess.run(
                    ["systemctl", "is-active", "clamav-daemon"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5
                )
                services["clamav"] = (result.returncode == 0)
                status = "ACTIVE" if services["clamav"] else "INACTIVE"
                self.log("INFO", f"ClamAV daemon: {status}")
            except Exception as e:
                self.log("WARNING", f"Could not check ClamAV status: {e}")
            
            # Check UFW
            try:
                result = subprocess.run(
                    ["ufw", "status"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5
                )
                output = result.stdout.decode('utf-8')
                services["ufw"] = ("Status: active" in output)
                services["firewall_active"] = services["ufw"]
                status = "ACTIVE" if services["ufw"] else "INACTIVE"
                self.log("INFO", f"UFW firewall: {status}")
            except Exception as e:
                self.log("WARNING", f"Could not check UFW status: {e}")
        
        return services
    
    def check_pandora_files(self) -> Dict[str, Any]:
        """Check if Pandora AIOS files are present"""
        self.log("INFO", "Checking Pandora AIOS files...")
        
        files = {
            "root_files": {},
            "extracted_files": {},
            "missing": []
        }
        
        # Check root files - try to find the repository root
        root_path = os.path.dirname(os.path.abspath(__file__))
        
        # Walk up to find repository root (contains .git)
        current = root_path
        for _ in range(5):  # Max 5 levels up
            if os.path.exists(os.path.join(current, '.git')):
                root_path = current
                break
            current = os.path.dirname(current)
        
        self.log("INFO", f"Repository root: {root_path}")
        
        expected_root = [
            "README.md",
            "PANDORA25.zip",
            "automerge_script.py",
            "quantum_virtual_processor.py",
            "self_learning_agent.py",
            "gemini_script_content.md",
            "diagnostic_system.py"
        ]
        
        for filename in expected_root:
            filepath = os.path.join(root_path, filename)
            exists = os.path.exists(filepath)
            files["root_files"][filename] = "present" if exists else "missing"
            if not exists:
                files["missing"].append(f"root:{filename}")
                self.log("WARNING", f"File '{filename}': NOT FOUND at {filepath}")
            else:
                self.log("INFO", f"File '{filename}': OK")
        
        return files
    
    def check_health_logs(self) -> Dict[str, Any]:
        """Check for health log files and their status"""
        self.log("INFO", "Checking health logs...")
        
        logs = {
            "logs_found": {},
            "errors_detected": {}
        }
        
        expected_logs = [
            "/var/log/pandora_supervisor.log",
            "/var/log/pandora_quantum_firewall.log",
            "/var/log/pandora_avfirewall.log",
            "/var/log/pandora_security.log",
            "/var/log/pandora_antivirus.log"
        ]
        
        for log_path in expected_logs:
            if os.path.exists(log_path):
                logs["logs_found"][log_path] = True
                self.log("INFO", f"Log file found: {log_path}")
                
                # Check for errors
                try:
                    with open(log_path, "r") as f:
                        content = f.read().lower()
                        error_keywords = ["critical", "fail", "panic", "error"]
                        errors = [kw for kw in error_keywords if kw in content]
                        if errors:
                            logs["errors_detected"][log_path] = errors
                            self.log("WARNING", f"Errors detected in {log_path}: {errors}")
                except Exception as e:
                    self.log("WARNING", f"Could not read {log_path}: {e}")
            else:
                logs["logs_found"][log_path] = False
                self.log("INFO", f"Log file not found (may be OK if system not started): {log_path}")
        
        return logs
    
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run complete diagnostic suite"""
        self.log("INFO", "=" * 60)
        self.log("INFO", "Starting Pandora AIOS Full Diagnostic")
        self.log("INFO", "=" * 60)
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_info": self.check_system_info(),
            "hardware": self.check_hardware(),
            "nvidia_gpu": self.check_nvidia_gpu(),
            "intel_cpu": self.check_intel_cpu(),
            "dependencies": self.check_dependencies(),
            "security_services": self.check_security_services(),
            "pandora_files": self.check_pandora_files(),
            "health_logs": self.check_health_logs(),
            "runtime_seconds": time.time() - self.start_time
        }
        
        # Summary
        self.log("INFO", "=" * 60)
        self.log("INFO", "Diagnostic Summary")
        self.log("INFO", "=" * 60)
        
        # Check for critical issues
        critical_issues = []
        warnings = []
        
        if not report["nvidia_gpu"]["available"]:
            critical_issues.append("NVIDIA GPU not detected or not functional")
        
        if not report["intel_cpu"]["is_intel"]:
            warnings.append("Intel CPU not detected (system may still work)")
        
        if report["dependencies"]["missing"]:
            warnings.append(f"{len(report['dependencies']['missing'])} dependencies missing")
        
        if not report["security_services"]["firewall_active"]:
            warnings.append("Firewall not active")
        
        if not report["security_services"]["clamav"]:
            warnings.append("ClamAV antivirus not running")
        
        if critical_issues:
            self.log("ERROR", f"Critical Issues ({len(critical_issues)}):")
            for issue in critical_issues:
                self.log("ERROR", f"  - {issue}")
        
        if warnings:
            self.log("WARNING", f"Warnings ({len(warnings)}):")
            for warning in warnings:
                self.log("WARNING", f"  - {warning}")
        
        if not critical_issues and not warnings:
            self.log("INFO", "System status: HEALTHY - All checks passed!")
        elif not critical_issues:
            self.log("INFO", "System status: OPERATIONAL - Some warnings present")
        else:
            self.log("ERROR", "System status: DEGRADED - Critical issues detected")
        
        self.log("INFO", f"Diagnostic completed in {report['runtime_seconds']:.2f} seconds")
        self.log("INFO", "=" * 60)
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: Optional[str] = None):
        """Save diagnostic report to JSON file"""
        if output_path is None:
            output_path = f"/tmp/pandora_diagnostic_report_{int(time.time())}.json"
        
        try:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            self.log("INFO", f"Report saved to: {output_path}")
            return output_path
        except Exception as e:
            self.log("ERROR", f"Could not save report: {e}")
            return None
    
    def continuous_monitoring(self, interval: int = 30, duration: int = 300):
        """Run continuous monitoring for specified duration"""
        self.log("INFO", f"Starting continuous monitoring (interval: {interval}s, duration: {duration}s)")
        
        start = time.time()
        iteration = 0
        
        while (time.time() - start) < duration:
            iteration += 1
            self.log("INFO", f"--- Monitoring iteration {iteration} ---")
            
            # Quick checks
            if psutil:
                cpu = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory()
                
                self.log("INFO", f"CPU: {cpu}%, Memory: {mem.percent}% ({mem.available / (1024**3):.2f}GB free)")
                
                if cpu > 90:
                    self.log("WARNING", f"High CPU usage detected: {cpu}%")
                
                if mem.percent > 90:
                    self.log("WARNING", f"High memory usage detected: {mem.percent}%")
            
            time.sleep(interval)
        
        self.log("INFO", "Continuous monitoring completed")


def main():
    """Main entry point for diagnostic system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pandora AIOS Diagnostic System")
    parser.add_argument("--full", action="store_true", help="Run full diagnostic suite")
    parser.add_argument("--monitor", action="store_true", help="Run continuous monitoring")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval (seconds)")
    parser.add_argument("--duration", type=int, default=300, help="Monitoring duration (seconds)")
    parser.add_argument("--output", type=str, help="Output file for report (JSON)")
    parser.add_argument("--log", type=str, default="/var/log/pandora_diagnostics.log", help="Log file path")
    
    args = parser.parse_args()
    
    diagnostics = PandoraDiagnostics(log_path=args.log)
    
    if args.monitor:
        diagnostics.continuous_monitoring(interval=args.interval, duration=args.duration)
    else:
        # Run full diagnostic by default
        report = diagnostics.run_full_diagnostic()
        
        if args.output:
            diagnostics.save_report(report, args.output)
        else:
            # Save to temp location
            output_path = diagnostics.save_report(report)
            print(f"\nFull report available at: {output_path}")


if __name__ == "__main__":
    main()
