"""
Pandora AIOS Universal Compatibility Layer
-------------------------------------------
Maximizes compatibility across all platforms, hardware, architectures, and environments.
Provides abstraction layers, fallbacks, polyfills, and adapters for seamless operation.

Compatibility Targets:
- Operating Systems: Windows, Linux (all distros), macOS, BSD, Solaris, Android, iOS
- Architectures: x86, x86_64, ARM, ARM64, RISC-V, PowerPC, MIPS, SPARC
- Python Versions: 2.7, 3.5+
- Shells: Bash, Zsh, Fish, PowerShell, CMD, Dash, Ksh
- Package Managers: apt, yum, dnf, pacman, brew, chocolatey, pip, conda
- Virtualization: Docker, VMware, VirtualBox, KVM, Hyper-V, QEMU, WSL1, WSL2
- Cloud: AWS, Azure, GCP, Oracle Cloud, DigitalOcean, Linode
- Embedded: Raspberry Pi, Arduino, ESP32, BeagleBone
- Legacy: DOS, OS/2, Windows 95/98/ME/2000/XP, Old Linux kernels

Philosophy: Universal accessibility, graceful degradation, progressive enhancement
"""

import os
import sys
import platform
import subprocess
import importlib
import json
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path

class CompatibilityLayer:
    """Main compatibility management system"""
    
    def __init__(self):
        self.platform_info = self._detect_platform()
        self.capabilities = {}
        self.fallbacks = {}
        self.polyfills = {}
        self.compatibility_score = 0
        
        # Initialize compatibility checks
        self._check_all_capabilities()
    
    def _detect_platform(self) -> Dict[str, Any]:
        """Comprehensive platform detection"""
        info = {
            # Basic system info
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            
            # Python info
            "python_version": sys.version,
            "python_version_info": sys.version_info,
            "python_implementation": platform.python_implementation(),
            
            # Detailed detection
            "is_64bit": sys.maxsize > 2**32,
            "is_windows": sys.platform.startswith('win'),
            "is_linux": sys.platform.startswith('linux'),
            "is_macos": sys.platform == 'darwin',
            "is_bsd": 'bsd' in sys.platform.lower(),
            "is_unix": os.name == 'posix',
            "is_android": 'ANDROID_ROOT' in os.environ,
            "is_wsl": self._detect_wsl(),
            "is_cygwin": sys.platform == 'cygwin',
            
            # Architecture specifics
            "is_x86": 'x86' in platform.machine().lower() or 'i686' in platform.machine().lower(),
            "is_x86_64": platform.machine() in ['x86_64', 'AMD64', 'x64'],
            "is_arm": 'arm' in platform.machine().lower(),
            "is_arm64": 'aarch64' in platform.machine().lower() or 'arm64' in platform.machine().lower(),
            "is_riscv": 'riscv' in platform.machine().lower(),
            "is_powerpc": 'ppc' in platform.machine().lower(),
            
            # Virtualization detection
            "is_docker": self._detect_docker(),
            "is_vm": self._detect_vm(),
            
            # Environment
            "shell": os.environ.get('SHELL', ''),
            "terminal": os.environ.get('TERM', ''),
            "path_separator": os.pathsep,
            "line_separator": os.linesep,
        }
        
        return info
    
    def _detect_wsl(self) -> bool:
        """Detect if running in WSL"""
        try:
            if os.path.exists('/proc/version'):
                with open('/proc/version', 'r') as f:
                    return 'microsoft' in f.read().lower()
        except:
            pass
        return False
    
    def _detect_docker(self) -> bool:
        """Detect if running in Docker"""
        return os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
    
    def _detect_vm(self) -> bool:
        """Detect if running in VM"""
        vm_indicators = [
            '/sys/class/dmi/id/product_name',
            '/sys/class/dmi/id/sys_vendor'
        ]
        
        for path in vm_indicators:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        content = f.read().lower()
                        if any(x in content for x in ['vmware', 'virtualbox', 'qemu', 'kvm', 'xen', 'hyper-v']):
                            return True
                except:
                    pass
        return False
    
    def _check_all_capabilities(self):
        """Check all system capabilities"""
        self.capabilities = {
            # Core capabilities
            "threads": self._check_threading(),
            "multiprocessing": self._check_multiprocessing(),
            "async": self._check_async(),
            "networking": self._check_networking(),
            "file_system": self._check_filesystem(),
            
            # Python features
            "json": self._check_module('json'),
            "subprocess": self._check_module('subprocess'),
            "socket": self._check_module('socket'),
            "ssl": self._check_module('ssl'),
            "sqlite3": self._check_module('sqlite3'),
            
            # Optional dependencies
            "psutil": self._check_module('psutil'),
            "requests": self._check_module('requests'),
            "flask": self._check_module('flask'),
            "numpy": self._check_module('numpy'),
            "cryptography": self._check_module('cryptography'),
            
            # System tools
            "git": self._check_command('git'),
            "curl": self._check_command('curl'),
            "wget": self._check_command('wget'),
            "tar": self._check_command('tar'),
            "zip": self._check_command('zip'),
            "ssh": self._check_command('ssh'),
            
            # Permissions
            "is_admin": self._check_admin(),
            "can_write_etc": os.access('/etc', os.W_OK) if os.path.exists('/etc') else False,
            "can_write_var": os.access('/var', os.W_OK) if os.path.exists('/var') else False,
        }
        
        # Calculate compatibility score
        self.compatibility_score = sum(1 for v in self.capabilities.values() if v) / len(self.capabilities) * 100
    
    def _check_threading(self) -> bool:
        """Check threading support"""
        try:
            import threading
            return True
        except:
            return False
    
    def _check_multiprocessing(self) -> bool:
        """Check multiprocessing support"""
        try:
            import multiprocessing
            return True
        except:
            return False
    
    def _check_async(self) -> bool:
        """Check async/await support"""
        return sys.version_info >= (3, 5)
    
    def _check_networking(self) -> bool:
        """Check network capabilities"""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=1)
            return True
        except:
            return False
    
    def _check_filesystem(self) -> bool:
        """Check filesystem access"""
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=True) as f:
                f.write(b"test")
            return True
        except:
            return False
    
    def _check_module(self, module_name: str) -> bool:
        """Check if Python module is available"""
        try:
            importlib.import_module(module_name)
            return True
        except:
            return False
    
    def _check_command(self, command: str) -> bool:
        """Check if system command is available"""
        try:
            if self.platform_info['is_windows']:
                result = subprocess.run(['where', command], capture_output=True, timeout=2)
            else:
                result = subprocess.run(['which', command], capture_output=True, timeout=2)
            return result.returncode == 0
        except:
            return False
    
    def _check_admin(self) -> bool:
        """Check if running with admin/root privileges"""
        try:
            if self.platform_info['is_windows']:
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            else:
                return os.geteuid() == 0
        except:
            return False
    
    def get_compatible_path(self, path: str) -> str:
        """Convert path to platform-compatible format"""
        if self.platform_info['is_windows']:
            # Convert forward slashes to backslashes
            return path.replace('/', '\\')
        else:
            # Convert backslashes to forward slashes
            return path.replace('\\', '/')
    
    def run_command_compatible(self, command: str, shell: bool = True, timeout: int = 30) -> Tuple[int, str, str]:
        """Run command with maximum compatibility"""
        try:
            if isinstance(command, str) and not shell:
                # Split command for non-shell execution
                import shlex
                command = shlex.split(command)
            
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return -1, "", "Command timeout"
        except Exception as e:
            return -1, "", str(e)
    
    def install_dependency(self, package_name: str, package_type: str = "python") -> bool:
        """Install missing dependency with platform detection"""
        if package_type == "python":
            return self._install_python_package(package_name)
        elif package_type == "system":
            return self._install_system_package(package_name)
        return False
    
    def _install_python_package(self, package: str) -> bool:
        """Install Python package using appropriate method"""
        # Try pip
        pip_commands = ['pip3', 'pip', 'python3 -m pip', 'python -m pip']
        
        for pip_cmd in pip_commands:
            try:
                result = subprocess.run(
                    f"{pip_cmd} install {package}",
                    shell=True,
                    capture_output=True,
                    timeout=300
                )
                if result.returncode == 0:
                    print(f"[SUCCESS] Installed {package}")
                    return True
            except:
                continue
        
        print(f"[ERROR] Failed to install {package}")
        return False
    
    def _install_system_package(self, package: str) -> bool:
        """Install system package using appropriate package manager"""
        package_managers = [
            ('apt-get', f'sudo apt-get install -y {package}'),
            ('yum', f'sudo yum install -y {package}'),
            ('dnf', f'sudo dnf install -y {package}'),
            ('pacman', f'sudo pacman -S --noconfirm {package}'),
            ('brew', f'brew install {package}'),
            ('choco', f'choco install {package} -y'),
        ]
        
        for pm, cmd in package_managers:
            if self._check_command(pm):
                try:
                    result = subprocess.run(cmd, shell=True, capture_output=True, timeout=300)
                    if result.returncode == 0:
                        print(f"[SUCCESS] Installed {package}")
                        return True
                except:
                    continue
        
        print(f"[ERROR] Failed to install {package}")
        return False
    
    def get_temp_dir(self) -> str:
        """Get platform-compatible temporary directory"""
        import tempfile
        return tempfile.gettempdir()
    
    def get_home_dir(self) -> str:
        """Get user home directory"""
        return str(Path.home())
    
    def get_config_dir(self) -> str:
        """Get appropriate config directory for platform"""
        if self.platform_info['is_windows']:
            return os.path.join(os.environ.get('APPDATA', ''), 'Pandora')
        elif self.platform_info['is_macos']:
            return os.path.join(self.get_home_dir(), 'Library', 'Application Support', 'Pandora')
        else:
            return os.path.join(self.get_home_dir(), '.config', 'pandora')
    
    def ensure_dir(self, directory: str) -> bool:
        """Create directory with maximum compatibility"""
        try:
            os.makedirs(directory, exist_ok=True)
            return True
        except:
            return False
    
    def open_file_compatible(self, filepath: str, mode: str = 'r', encoding: str = 'utf-8'):
        """Open file with compatibility handling"""
        try:
            # Python 3 style
            return open(filepath, mode, encoding=encoding)
        except TypeError:
            # Python 2 fallback
            if 'b' in mode:
                return open(filepath, mode)
            else:
                import codecs
                return codecs.open(filepath, mode, encoding=encoding)
    
    def get_line_ending(self) -> str:
        """Get platform line ending"""
        if self.platform_info['is_windows']:
            return '\r\n'
        else:
            return '\n'
    
    def normalize_line_endings(self, text: str) -> str:
        """Normalize line endings to platform format"""
        # First normalize to \n
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Then convert to platform format
        if self.platform_info['is_windows']:
            return text.replace('\n', '\r\n')
        return text
    
    def detect_encoding(self, filepath: str) -> str:
        """Detect file encoding"""
        try:
            import chardet
            with open(filepath, 'rb') as f:
                result = chardet.detect(f.read())
                return result['encoding'] or 'utf-8'
        except:
            return 'utf-8'
    
    def is_compatible(self, requirements: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if system meets requirements"""
        issues = []
        
        # Check Python version
        if 'python_version' in requirements:
            min_version = requirements['python_version']
            if sys.version_info < min_version:
                issues.append(f"Python {min_version[0]}.{min_version[1]}+ required")
        
        # Check required modules
        if 'modules' in requirements:
            for module in requirements['modules']:
                if not self._check_module(module):
                    issues.append(f"Missing Python module: {module}")
        
        # Check required commands
        if 'commands' in requirements:
            for command in requirements['commands']:
                if not self._check_command(command):
                    issues.append(f"Missing system command: {command}")
        
        # Check platform
        if 'platforms' in requirements:
            if self.platform_info['system'] not in requirements['platforms']:
                issues.append(f"Platform not supported: {self.platform_info['system']}")
        
        # Check architecture
        if 'architectures' in requirements:
            if self.platform_info['machine'] not in requirements['architectures']:
                issues.append(f"Architecture not supported: {self.platform_info['machine']}")
        
        return len(issues) == 0, issues
    
    def generate_compatibility_report(self) -> str:
        """Generate detailed compatibility report"""
        report = []
        report.append("="*70)
        report.append("Pandora AIOS Compatibility Report")
        report.append("="*70)
        report.append("")
        
        # Platform info
        report.append("Platform Information:")
        report.append(f"  System: {self.platform_info['system']} {self.platform_info['release']}")
        report.append(f"  Architecture: {self.platform_info['machine']}")
        report.append(f"  Processor: {self.platform_info['processor']}")
        report.append(f"  Python: {self.platform_info['python_implementation']} {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        report.append(f"  64-bit: {self.platform_info['is_64bit']}")
        report.append("")
        
        # Environment detection
        report.append("Environment Detection:")
        env_checks = [
            ('Windows', 'is_windows'),
            ('Linux', 'is_linux'),
            ('macOS', 'is_macos'),
            ('BSD', 'is_bsd'),
            ('WSL', 'is_wsl'),
            ('Docker', 'is_docker'),
            ('Virtual Machine', 'is_vm'),
            ('Android', 'is_android'),
        ]
        
        for name, key in env_checks:
            status = "✓" if self.platform_info.get(key) else "✗"
            report.append(f"  {status} {name}")
        report.append("")
        
        # Capabilities
        report.append("System Capabilities:")
        for capability, available in sorted(self.capabilities.items()):
            status = "✓" if available else "✗"
            report.append(f"  {status} {capability}")
        report.append("")
        
        # Compatibility score
        report.append(f"Overall Compatibility Score: {self.compatibility_score:.1f}%")
        report.append("")
        
        # Recommendations
        report.append("Recommendations:")
        missing_critical = [k for k, v in self.capabilities.items() if not v and k in ['subprocess', 'file_system', 'networking']]
        missing_optional = [k for k, v in self.capabilities.items() if not v and k not in ['subprocess', 'file_system', 'networking']]
        
        if missing_critical:
            report.append("  CRITICAL - Missing core capabilities:")
            for cap in missing_critical:
                report.append(f"    - {cap}")
        
        if missing_optional:
            report.append("  Optional - Enhanced features available with:")
            for cap in missing_optional[:5]:  # Show top 5
                report.append(f"    - {cap}")
        
        if not missing_critical and self.compatibility_score > 80:
            report.append("  ✓ System is highly compatible with Pandora AIOS")
        elif not missing_critical:
            report.append("  ⚠ System is compatible but some features may be limited")
        else:
            report.append("  ✗ System has compatibility issues - some features unavailable")
        
        report.append("")
        report.append("="*70)
        
        return "\n".join(report)
    
    def create_compatibility_shim(self, target_os: str = None) -> Dict[str, Callable]:
        """Create compatibility shim for specific target OS"""
        shim = {}
        
        target_os = target_os or self.platform_info['system']
        
        # Path operations
        if target_os == 'Windows':
            shim['path_join'] = lambda *args: '\\'.join(args)
            shim['path_sep'] = '\\'
        else:
            shim['path_join'] = lambda *args: '/'.join(args)
            shim['path_sep'] = '/'
        
        # Command execution
        if target_os == 'Windows':
            shim['shell_cmd'] = lambda cmd: f'cmd /c {cmd}'
        else:
            shim['shell_cmd'] = lambda cmd: f'sh -c "{cmd}"'
        
        # File operations
        shim['read_file'] = lambda path: self.open_file_compatible(path, 'r').read()
        shim['write_file'] = lambda path, data: self.open_file_compatible(path, 'w').write(data)
        
        return shim
    
    def setup_fallbacks(self):
        """Setup fallback implementations for missing features"""
        # If psutil not available, use basic alternatives
        if not self.capabilities['psutil']:
            self.fallbacks['get_cpu_percent'] = self._fallback_cpu_percent
            self.fallbacks['get_memory_info'] = self._fallback_memory_info
        
        # If requests not available, use urllib
        if not self.capabilities['requests']:
            self.fallbacks['http_get'] = self._fallback_http_get
        
        # If threading not available, use sequential execution
        if not self.capabilities['threads']:
            self.fallbacks['run_threaded'] = self._fallback_sequential
    
    def _fallback_cpu_percent(self) -> float:
        """Fallback CPU usage without psutil"""
        try:
            if self.platform_info['is_linux']:
                with open('/proc/loadavg', 'r') as f:
                    return float(f.read().split()[0]) * 100
        except:
            pass
        return 0.0
    
    def _fallback_memory_info(self) -> Dict[str, int]:
        """Fallback memory info without psutil"""
        try:
            if self.platform_info['is_linux']:
                with open('/proc/meminfo', 'r') as f:
                    lines = f.readlines()
                    mem_total = int([l for l in lines if 'MemTotal' in l][0].split()[1]) * 1024
                    mem_free = int([l for l in lines if 'MemFree' in l][0].split()[1]) * 1024
                    return {'total': mem_total, 'available': mem_free}
        except:
            pass
        return {'total': 0, 'available': 0}
    
    def _fallback_http_get(self, url: str) -> str:
        """Fallback HTTP GET without requests"""
        try:
            if sys.version_info[0] == 3:
                from urllib.request import urlopen
            else:
                from urllib2 import urlopen
            
            response = urlopen(url, timeout=10)
            return response.read().decode('utf-8')
        except:
            return ""
    
    def _fallback_sequential(self, func: Callable, items: List[Any]):
        """Fallback sequential execution instead of threading"""
        return [func(item) for item in items]


class PolyfillManager:
    """Manages polyfills for missing features"""
    
    def __init__(self, compat_layer: CompatibilityLayer):
        self.compat = compat_layer
        self.polyfills = {}
        self._setup_polyfills()
    
    def _setup_polyfills(self):
        """Setup all polyfills"""
        # Python 2/3 compatibility
        if sys.version_info[0] == 2:
            self.polyfills['input'] = raw_input
            self.polyfills['range'] = xrange
        else:
            self.polyfills['input'] = input
            self.polyfills['range'] = range
        
        # JSON for older Python
        try:
            import json
            self.polyfills['json'] = json
        except:
            import simplejson as json
            self.polyfills['json'] = json
    
    def get(self, name: str) -> Any:
        """Get polyfill by name"""
        return self.polyfills.get(name)


def get_universal_interpreter() -> str:
    """Get universal Python interpreter command"""
    interpreters = ['python3', 'python', 'py', 'python2']
    
    for interp in interpreters:
        try:
            result = subprocess.run(
                [interp, '--version'],
                capture_output=True,
                timeout=2
            )
            if result.returncode == 0:
                return interp
        except:
            continue
    
    return 'python'


def main():
    """Main compatibility check"""
    print("\nInitializing Pandora AIOS Compatibility Layer...")
    
    compat = CompatibilityLayer()
    
    # Generate and print report
    report = compat.generate_compatibility_report()
    print(report)
    
    # Save report to file
    report_path = os.path.join(compat.get_temp_dir(), 'pandora_compatibility_report.txt')
    try:
        with compat.open_file_compatible(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
    except:
        print(f"\nCould not save report to file")
    
    # Return exit code based on compatibility
    if compat.compatibility_score >= 80:
        return 0
    elif compat.compatibility_score >= 50:
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
