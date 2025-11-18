"""
SystemMonitoring
Regular, transparent health/resource checks for the AIOS fabric.
Philosophy: Self-examination, transparency—reports state calmly.
"""
import time
import os

try:
    import psutil
except ImportError:
    psutil = None

class SystemMonitoring:
    def __init__(self, orchestrator=None, check_interval=5):
        self.orchestrator = orchestrator
        self.check_interval = check_interval
        self.alerts = []
        
    def get_cpu_usage(self):
        """Get current CPU usage percentage"""
        if psutil:
            return psutil.cpu_percent(interval=1)
        return 0
    
    def get_memory_usage(self):
        """Get memory usage statistics"""
        if psutil:
            mem = psutil.virtual_memory()
            return {
                'total_gb': round(mem.total / (1024**3), 2),
                'used_gb': round(mem.used / (1024**3), 2),
                'percent': mem.percent
            }
        return {'total_gb': 0, 'used_gb': 0, 'percent': 0}
    
    def get_disk_usage(self):
        """Get disk usage for root partition"""
        if psutil:
            disk = psutil.disk_usage('/')
            return {
                'total_gb': round(disk.total / (1024**3), 2),
                'used_gb': round(disk.used / (1024**3), 2),
                'percent': disk.percent
            }
        return {'total_gb': 0, 'used_gb': 0, 'percent': 0}
    
    def check_health(self):
        """Perform comprehensive health check"""
        health_report = {
            'timestamp': time.time(),
            'cpu': self.get_cpu_usage(),
            'memory': self.get_memory_usage(),
            'disk': self.get_disk_usage(),
            'status': 'healthy',
            'warnings': []
        }
        
        # Check for concerning conditions (stoic thresholds)
        if health_report['cpu'] > 85:
            health_report['warnings'].append("High CPU usage detected")
            health_report['status'] = 'warning'
            
        if health_report['memory']['percent'] > 85:
            health_report['warnings'].append("High memory usage detected")
            health_report['status'] = 'warning'
            
        if health_report['disk']['percent'] > 90:
            health_report['warnings'].append("Low disk space")
            health_report['status'] = 'warning'
        
        return health_report
        
    def run(self):
        """Periodically log and reflect on system well-being"""
        print("System monitoring started...")
        try:
            while True:
                report = self.check_health()
                self._print_report(report)
                
                # Notify orchestrator if present
                if self.orchestrator and hasattr(self.orchestrator, 'handle_health_report'):
                    self.orchestrator.handle_health_report(report)
                    
                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            print("\nSystem monitoring stopped.")
    
    def _print_report(self, report):
        """Print health report to console"""
        print(f"\n[{time.strftime('%H:%M:%S')}] System Health: {report['status'].upper()}")
        print(f"  CPU: {report['cpu']:.1f}%")
        print(f"  Memory: {report['memory']['used_gb']:.1f}GB / {report['memory']['total_gb']:.1f}GB ({report['memory']['percent']:.1f}%)")
        print(f"  Disk: {report['disk']['used_gb']:.1f}GB / {report['disk']['total_gb']:.1f}GB ({report['disk']['percent']:.1f}%)")
        if report['warnings']:
            print(f"  ⚠ Warnings: {', '.join(report['warnings'])}")

if __name__ == "__main__":
    monitor = SystemMonitoring(check_interval=3)
    monitor.run()
class SystemMonitoring:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
    def run(self):
        # Periodically log and reflect on system well-being
        pass
        pass
