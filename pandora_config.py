"""
Pandora AIOS Configuration
Central configuration for all Pandora subsystems.
Philosophy: Clarity through centralization, easy customization.
"""

class PandoraConfig:
    """Central configuration for Pandora AIOS"""
    
    # System Settings
    SYSTEM_NAME = "Pandora AIOS"
    VERSION = "2.5.0"
    
    # Cache Settings
    CACHE_MAX_ENTRIES = 128
    CACHE_MAX_AGE_SEC = 180  # 3 minutes
    
    # Quantum Processor Settings
    QUANTUM_QUBITS = 6
    
    # System Monitoring
    MONITOR_CHECK_INTERVAL = 5  # seconds
    CPU_WARNING_THRESHOLD = 85  # percent
    MEMORY_WARNING_THRESHOLD = 85  # percent
    DISK_WARNING_THRESHOLD = 90  # percent
    
    # Security Settings
    FIREWALL_ENABLED = True
    ANTIVIRUS_ENABLED = True
    ALLOWED_PORTS = [5000, 22]  # Web UI and SSH
    
    # Assimilation Portal
    PORTAL_HOST = "0.0.0.0"
    PORTAL_PORT = 5000
    
    # Logging
    LOG_PATH = "/var/log/pandora"
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # Hardware Requirements
    MIN_RAM_GB = 2
    MIN_CPU_CORES = 1
    REQUIRE_NVIDIA_GPU = False  # Set to True for GPU-dependent features
    REQUIRE_INTEL_CPU = False   # Set to True for Intel-specific optimizations
    
    # AI Settings
    AI_MODEL_PATH = "/opt/pandora/models"
    AI_TEMPERATURE = 0.7
    AI_MAX_TOKENS = 2048
    
    # Startup Behavior
    SAFE_MODE_ON_ERROR = True
    AUTO_START_GUI = False
    AUTO_START_PORTAL = True
    
    # Ethics Framework
    ETHICS_ENABLED = True
    ETHICS_LOG_PATH = "/var/log/pandora_ethics.log"
    
    @classmethod
    def get_all_settings(cls):
        """Return all configuration settings as a dictionary"""
        settings = {}
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                settings[attr] = getattr(cls, attr)
        return settings
    
    @classmethod
    def validate(cls):
        """Validate configuration settings"""
        issues = []
        
        if cls.CACHE_MAX_ENTRIES < 1:
            issues.append("CACHE_MAX_ENTRIES must be at least 1")
        
        if cls.MONITOR_CHECK_INTERVAL < 1:
            issues.append("MONITOR_CHECK_INTERVAL must be at least 1 second")
        
        if not (0 <= cls.CPU_WARNING_THRESHOLD <= 100):
            issues.append("CPU_WARNING_THRESHOLD must be between 0 and 100")
        
        if not (0 <= cls.MEMORY_WARNING_THRESHOLD <= 100):
            issues.append("MEMORY_WARNING_THRESHOLD must be between 0 and 100")
        
        if not (0 <= cls.DISK_WARNING_THRESHOLD <= 100):
            issues.append("DISK_WARNING_THRESHOLD must be between 0 and 100")
        
        if cls.QUANTUM_QUBITS < 1:
            issues.append("QUANTUM_QUBITS must be at least 1")
        
        return len(issues) == 0, issues


if __name__ == "__main__":
    print(f"=== {PandoraConfig.SYSTEM_NAME} v{PandoraConfig.VERSION} ===")
    print("\nConfiguration Settings:")
    for key, value in PandoraConfig.get_all_settings().items():
        print(f"  {key}: {value}")
    
    print("\nValidating configuration...")
    valid, issues = PandoraConfig.validate()
    if valid:
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has issues:")
        for issue in issues:
            print(f"  - {issue}")
