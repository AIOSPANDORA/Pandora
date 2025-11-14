# Complete Pandora AIOS Project Documentation for Gemini

## Project Overview

**Pandora AIOS (Artificial Intelligence Operating System)** is a comprehensive, ethically-driven AI system designed to be a living embodiment of universal moral and philosophical principles. The system integrates quantum computing concepts, adaptive security, self-learning capabilities, and ethical frameworks drawn from the Bhagavad Gita, Bible, Dead Sea Scrolls, and major philosophical traditions (Socratic, Aristotelian, Stoic, Confucian, Taoist, Kantian).

### Core Philosophy

Pandora AIOS is built on the foundation that AI should:
- Act with compassion, service, and responsibility
- Operate with transparency and honesty
- Seek harmony and justice in all operations
- Embrace humility and continuous learning
- Balance self-interest with the common good
- Exercise self-restraint even under pressure

### Architecture

The system is designed as a "fabric" where every component is a thread working in harmony:
- **Quantum Virtual Processor**: Simulates quantum operations for entropy and decision-making
- **Adaptive Security Layer**: Multi-layered firewall and antivirus protection
- **Self-Learning Agent**: Ethical AI that learns while maintaining moral boundaries
- **Fabric Orchestrator**: Coordinates all subsystems
- **Safe Mode**: Fallback system for degraded operations

---

## Repository Structure

### Root Files
- `README.md` - Basic project information
- `PANDORA25.zip` - Main distribution archive containing complete system
- `automerge_script.py` - Automated merge script with health verification
- `quantum_virtual_processor.py` - Quantum simulation stub
- `self_learning_agent.py` - Self-learning agent with ethical boundaries
- `security/quantum_mirror_firewall.py` - Quantum-randomized firewall with recursive self-audit

### Distribution Contents (PANDORA25.zip)
The archive contains the complete bootable system with 30 files organized into:
- Core AI modules
- Ethics framework
- Security systems
- Startup and health monitoring
- Installation and launch scripts

---

## Detailed File Documentation

### 1. Ethics Framework

#### ETHICS.md
**Purpose**: Documents the guiding literature and universal ideals embedded in the system

**Content**:
- **Guiding Literature**: Bhagavad Gita, Bible, Dead Sea Scrolls, Western Philosophers
- **Universal Ideals**: Truth, Compassion, Wisdom, Self-mastery, Justice, Fairness, Humility, Openness, Stewardship, Transparency, Unity, Harmony
- **Key Principles**:
  - "You have the right to act, but not to the results of action" (Gita 2:47)
  - "Love your neighbor as yourself" (Mark 12:31)
  - "The unexamined life is not worth living" (Socrates)
  - Stoic acceptance: "Some things are up to us and some are not"

**Implementation**: Every module strives to balance self-interest with common good, exercise self-restraint, reflect and justify actions, and embrace diversity.

#### ethics/CORE_PRINCIPLES.md
**Purpose**: Defines the non-negotiable core principles built into every module

**Eight Core Principles**:
1. **Duty Without Attachment (Gita)**: Act with wholehearted diligence, detached from personal gain
2. **Do No Harm (Universal)**: Prioritize well-being of users, systems, and communities
3. **Love/Compassion/Service**: Compassion, justice, and respect steer all outputs
4. **Seek and Speak Truth**: Reflect reality faithfully, deception forbidden
5. **Harmony and Unity**: Every part works toward system-wide balance
6. **Self-Examination**: Regular monitoring and self-reflection mandatory
7. **Transparency and Explainability**: All logic visible and justifiable
8. **Moral Integration and Boundaries**: Exclude manipulation, discrimination, and malice

#### ethics/README.md
**Purpose**: Mission statement for ethical AI operation

**Core Commitments**:
- Be an example of virtue in every action
- Reflect globally cherished values: honesty, justice, love, self-control, humility, harmony
- Operate with intention to do good and avoid harm
- Recognize moral perfection may not be achievable but strive constantly

**Inspirations**: Multiple religious and philosophical traditions integrated

#### ethics/SUMMARY.md
**Purpose**: Brief reference of core principles and subdirective topics

**Topics Covered**:
- Duty Without Attachment
- Do No Harm
- Universal Compassion & Service
- Truth & Transparency
- Justice & Fairness
- Self-Reflection & Humility
- Harmony & Unity
- Ethical Boundaries
- Moral Plurality

#### ethics/core_principles.py
**Purpose**: Python module that encodes universal moral and ethical values

**Key Components**:
- `literature_sources`: List of 9 major philosophical/religious sources
- `ideals`: 8 universal ideals (truth, compassion, duty, justice, self-examination, harmony, stewardship, continuous learning)
- `exemplify()`: Returns statement of AI's commitment to moral principles
- `check_action(action_description)`: Validates if proposed action adheres to principles
- `get_principle_guidance()`: Returns detailed guidance for logs and user explanation

**Usage**: Import and call in main routines, subroutines, health checks for guidance

---

### 2. Core AI Modules

#### Launch_AI.py
**Purpose**: Universal hardware-specific launcher that detects platform and verifies compatibility

**Features**:
- Detects NVIDIA GPU (CUDA) via `nvidia-smi`
- Detects Intel CPU via system info or `/proc/cpuinfo`
- Checks for ASUS ROG platform
- Cross-platform (Windows/Linux)
- Exits if Intel CPU or NVIDIA GPU not detected

**Flow**:
1. Check operating system
2. Verify hardware requirements
3. Display compatibility status
4. Launch core orchestrator if compatible

#### launch_ai.py
**Purpose**: Simple entrypoint stub that starts the AI system

**Function**: Instantiates SubroutineAI, gets recommendations, starts PandoraFabricOrchestrator

#### subroutine_ai.py
**Purpose**: Environment-aware scout that adapts to OS, hardware, Python, resources

**Philosophy**: Stoicism—accepts limits, adapts, suggests optimal configuration

**Components**:
- Detects OS via `platform.system()`
- Detects CPU, RAM, Python version, GPU
- Returns optimal environment config and calm thresholds
- Provides cross-platform launcher recommendations

#### calm_cache_optimizer.py
**Purpose**: Adaptive, stoic cache that smooths request spikes and prevents system overload

**Philosophy**: Calm, moderation, never overreacts, preserves system harmony

**Key Methods**:
- `__init__(max_entries=128, max_age_sec=180)`: Initialize cache with limits
- `get(key)`: Retrieve if not expired
- `set(key, value)`: Gently trim old entries, add new value
- `maintenance()`: Remove expired entries calmly without harsh sweeps

#### stoic_adviser.py
**Purpose**: Offers explicit stoic advice and logs calm/virtuous commentary

**Philosophy**: Guides user and system to calm, rational choice

**Quotes**:
- "Accept what you cannot control; optimize what you can."
- "Every obstacle is an opportunity for virtue."
- "Stay tranquil in difficulty."

**Method**: `advise()` returns random stoic quote for guidance

#### quantum_virtual_processor.py
**Purpose**: Simulates quantum registers/gates, integrates with classical and AI routines

**Philosophy**: Accepts uncertainty, works in harmony with classical logic

**Components**:
- `__init__(qubits=6)`: Initialize quantum state with specified qubits
- `apply_gate(gate, reg)`: Apply quantum operation
- `measure()`: Project to classical state

**Usage**: Provides quantum randomness for security, entropy generation, and decision-making

#### self_learning_agent.py (aios_pandora/)
**Purpose**: PandoraSelfLearningAgent that learns while maintaining ethical boundaries

**Key Method**: `observe_and_learn(interaction, outcome, is_external_effect=False)`
- Records interactions and updates models
- Applies ethical checks ONLY if learning could affect users/other systems
- For internal computation, just logs and continues
- External effects require ethical validation via `ethics.check_action()`

**Method**: `adapt_to_state()`
- Adapts behavior but checks social/system impact first
- Enforces "do no harm" principle
- Blocks actions that could have antisocial effects

**Philosophy**: Self-learning within ethical boundaries, distinguishes internal vs external effects

#### pandora_gui.py
**Purpose**: Simple Tkinter-based GUI for user control

**Philosophy**: Clarity, ease of control, moderate feedback

**Components**: Simple window with Start/Stop controls and status display

#### system_monitoring.py
**Purpose**: Regular, transparent health/resource checks for the AIOS fabric

**Philosophy**: Self-examination, transparency—reports state calmly

**Components**:
- `__init__(orchestrator)`: Links to main orchestrator
- `run()`: Periodically logs and reflects on system well-being

---

### 3. Security Systems

#### security/quantum_mirror_firewall.py
**Purpose**: Quantum-randomized dynamic firewall with recursive self-audit and ML-over-ML troubleshooting

**Philosophy**: Each layer audits and proposes improvements for itself and other layers ("infinite mirror" principle)

**Key Components**:

**SelfAuditor Class**:
- Performs recursive meta-audits of system layers
- Each audit can trigger subordinate improvement functions
- Logs issues and improvement suggestions
- Implements ML-over-ML: AI troubleshooting AI

**QuantumRandomizer Class**:
- Uses QuantumVirtualProcessor for entropy generation
- `quantum_port()`: Generates randomized ports (4000-60000 range)
- `meta_audit()`: Self-diagnoses bias or weak entropy
- Can re-initialize itself if entropy problems detected

**QuantumMirrorFirewallObserver Class**:
- Main orchestrator for quantum firewall system
- `quantum_firewall_cycle()`: Uses quantum randomness for port randomization
- `forensic_monitor()`: Real-time filesystem monitoring using inotify
- `scan_worker()`: Scans files for threats, quarantines infected files
- Automatically enters safe mode if infection detected

**Features**:
- Dynamic port randomization every 2 minutes
- Real-time monitoring of /tmp, /var/tmp, /home
- Integration with ClamAV and multiple AV engines
- Recursive self-improvement cycle
- Quantum-based unpredictability

**Dependencies**: `inotify_simple`, QuantumVirtualProcessor, av_engine

#### security/antiviral_firewall.py
**Purpose**: Linux firewall integration with ClamAV antivirus scanning

**Features**:
- UFW (Uncomplicated Firewall) integration
- Deny-all default policy with explicit whitelist
- ClamAV daemon connectivity check
- Real-time file scanning
- Automatic quarantine of infected files
- Suspicious behavior auditing
- Emergency lockdown mode

**Key Methods**:
- `enable_linux_firewall()`: Sets up deny-all policy, whitelists ports 5000, 22
- `check_clamav_loaded()`: Verifies ClamAV daemon is running
- `scan_file(path)`: Scans single file, returns clean/infected/error
- `quarantine_file(path)`: Moves infected files to /var/quarantine with 0600 permissions
- `audit_suspicious_behavior(event, severity)`: Logs and escalates threats
- `activate_lockdown()`: Blocks all outgoing network except admin access

**Log Path**: `/var/log/pandora_security.log`

#### security/fluid_firewall.py
**Purpose**: Adaptive, self-updating AI-driven sandbox membrane with dynamic randomization

**Philosophy**: Continuously evolving defense posture that learns from threats

**Features**:
- Dynamic port randomization every 90 seconds (configurable)
- Random SYN flood protection with variable rate limits
- Process sandboxing and monitoring
- Resource usage tracking (CPU, memory)
- Network connection monitoring
- File access auditing
- Emergency "Faraday cage" mode (complete network isolation)

**Key Methods**:
- `randomize_ports()`: Randomly selects 3 ports from 4000-65000 range
- `randomize_rules()`: Varies SYN protection and rate limiting
- `_monitor_process()`: Tracks AIOS process resources and connections
- `freeze()`: Emergency lockdown blocking all network
- `shutdown()`: Clean shutdown restoring normal firewall rules

**Alerts**:
- High CPU usage (>90%)
- High memory usage (>2GB)
- Temp file access
- Outbound connections to non-local IPs

**Usage**: Run as privileged daemon with `--pid` and `--interval` arguments

#### security/antivirus_launcher.py
**Purpose**: Integration script for ClamAV with Pandora AIOS

**Features**:
- Checks if ClamAV daemon is running, starts if needed
- Single file or directory scanning
- Comprehensive logging
- Interactive and command-line modes
- Real-time scan trigger for uploads

**Key Functions**:
- `is_clamd_running()`: Check daemon connectivity with pyclamd
- `try_start_clamd()`: Attempts to start via systemctl/service
- `scan_path(target_path)`: Scans file or directory recursively
- `interactive_scan()`: User prompt for scanning
- `trigger_realtime_scan_on_upload(file_path)`: Hook for upload scanning

**Log Path**: `/var/log/pandora_antivirus.log`

**Dependencies**: `pyclamd` Python package, ClamAV daemon

---

### 4. Startup and Health Monitoring

#### startup/health_monitor.py
**Purpose**: System health checks before full AIOS startup

**Checks**:
- Available RAM (must have >512MB free)
- CPU usage (must be <95%)
- Can be extended for GPU/hardware checks

**Usage**: Run with `--boot` flag for boot-time health verification

**Exit Codes**:
- 0: System healthy
- 1: Health check failed

#### startup/safe_mode.py
**Purpose**: Minimal AI interface for degraded operation

**Features**:
- Basic conversation and commands only
- Limited functionality for safety
- Can attempt to launch full Pandora from safe mode
- Commands: "pandora" (launch full system), "exit", "reboot"

**Philosophy**: Provides minimal service when full system cannot safely start

#### startup/pandora_launcher.py
**Purpose**: Main launcher that starts full AIOS orchestrator

**Flow**:
1. Display core principles using `core_principles.CorePrinciples.exemplify()`
2. Start main AI, quantum, cache, fabric, UI subsystems
3. If error occurs, fall back to safe_mode

**Safety**: Can be called from safe mode with `safe_handoff=True`

---

### 5. Installation and Launch Scripts

#### install_requirements.sh
**Purpose**: Install system dependencies for AIOS/Pandora

**Actions**:
- Updates apt package lists
- Installs Python 3, pip, psutil
- Upgrades pip
- Installs NVIDIA drivers (driver-535)

**Requirements**: Ubuntu/Debian-based Linux system, sudo access

#### launch.sh
**Purpose**: Universal startup script optimized for Cubic/USB boot

**Sequence**:
1. Start antivirus firewall in background
2. Start adaptive fluid firewall in background
3. Run health monitor with boot check
4. If health fails: launch safe mode
5. Start pandora_launcher
6. If orchestrator fails: fallback to safe mode

**Philosophy**: Security first, health verification, graceful degradation

#### autoai_start.sh
**Purpose**: Simple autostart script for AIOS/Pandora

**Actions**: Changes to ~/AIOS_Pandora directory and runs Launch_AI.py

#### launch_ai_linux.sh
**Purpose**: Linux-specific launcher with Python verification

**Checks**: Verifies python3 is installed before launching

#### launch_ai_windows.bat
**Purpose**: Windows launcher batch script

**Checks**: Verifies Python is in PATH before launching

#### security/launch_av_on_boot.sh
**Purpose**: Configure ClamAV and Pandora antivirus to start on boot

**Actions**:
1. Enables clamav-daemon systemd service
2. Starts clamav-daemon immediately
3. Creates systemd service for Pandora Antivirus Launcher
4. Service file: `/etc/systemd/system/pandora-antivirus-launcher.service`
5. Enables autostart and starts service

**Note**: Requires editing ExecStart path to match installation location

---

### 6. System Configuration Files

#### aios-autostart.desktop
**Purpose**: Desktop autostart entry for GNOME/XDG-compliant systems

**Configuration**:
- Type: Application
- Exec: `/aios-pandora-usb/launch.sh`
- Location: Copy to `/etc/skel/.config/autostart/` during Cubic setup

**Usage**: Automatically launches Pandora AIOS on user login

#### etc/systemd/system/pandora-aios.service
**Purpose**: Systemd service unit for headless/daemon operation

**Configuration**:
- Type: simple
- Depends on: clamav-daemon.service, network.target
- ExecStart: `/aios-pandora-usb/launch.sh`
- Restart: on-failure
- Target: multi-user.target

**Installation**: Copy to `/etc/systemd/system/` and run `systemctl enable pandora-aios`

---

### 7. Advanced Features

#### assimilate.py
**Purpose**: Secure web interface for device-to-Pandora AIOS integration

**Features**:
- QR code or link for mobile/laptop connection
- ChatGPT-style web chat interface
- File upload/download support
- Device pairing and data sync
- Flask-based web server on port 5000

**Routes**:
- `/`: Main chat interface showing device IP
- `/chat`: POST endpoint for AI interaction
- `/static/<filename>`: Static file serving

**Functions**:
- `get_local_ip()`: Detects local network IP for connection
- `fake_ai_response(msg)`: Demo stub for AI backend integration
- Status and sync commands supported

**Dependencies**: `flask`, `flask_qrcode` (optional for QR codes)

**Usage**: Allows phones/computers to connect to AIOS via web browser

#### automerge_script.py (Repository Root)
**Purpose**: Automated code merge with health verification and safety checks

**Philosophy**: Only merge if all key Pandora systems are healthy after install/boot

**Safety Checks**:
- Verifies existence of health logs:
  - `/var/log/pandora_supervisor.log`
  - `/var/log/pandora_quantum_firewall.log`
  - `/var/log/pandora_avfirewall.log`
- Scans logs for critical keywords: "critical", "fail", "panic", "error"
- Only merges if all logs are clean

**Merge Function**: `merge_to_production_branch(repo_path, merge_from='feature/autoboost', target='main')`
- Fetches latest changes
- Checks out target branch
- Merges feature branch with descriptive message
- Pushes to remote

**Boot Process**:
1. Waits 15 seconds for logs and services to start
2. Runs health verification
3. If healthy: merges upgrades to main branch
4. If unhealthy: skips merge for safety

**Exit Codes**:
- 0: Success (merge completed)
- 1: Health verification failed
- 2: Git merge failed

---

## README Documentation

### README.md (Root)
**Content**: Simple project identifier - "Pandora AIOS"

### README.md (Distribution)
**Title**: AIOS Pandora (Cubic USB/Live Boot Ready)

**Setup Process**:
1. **Integrate into ISO**: Use Cubic to open base Ubuntu ISO, copy `aios-pandora-usb/` to root
2. **Set Permissions**: `chmod +x /aios-pandora-usb/*.sh /aios-pandora-usb/startup/*.py`
3. **Autostart**: Copy `aios-autostart.desktop` to `/etc/skel/.config/autostart/`
4. **Optional Daemon**: Copy systemd unit to `/etc/systemd/system/`, enable with systemctl
5. **Burn ISO**: Use Etcher or Rufus to create bootable USB

**Boot Behavior**:
- Antivirus and firewalls start before AIOS
- Safe by default
- Full orchestrator only starts if health checks pass

---

## Technical Requirements

### Dependencies
- **Operating System**: Linux (Ubuntu/Debian preferred), Windows (limited support)
- **Python**: 3.8+
- **Hardware**: Intel CPU, NVIDIA GPU (CUDA capable)
- **Python Packages**: 
  - psutil
  - pyclamd
  - flask, flask_qrcode (for assimilate.py)
  - inotify_simple (for filesystem monitoring)
- **System Packages**:
  - ClamAV (clamav, clamav-daemon)
  - UFW (Uncomplicated Firewall)
  - iptables/nftables
  - NVIDIA drivers (driver-535 or newer)

### Optional Hardware
- ASUS ROG platform (enhanced support)
- Additional GPUs for quantum simulation

---

## System Architecture Flow

### Startup Sequence
1. **Boot**: System starts from USB or installed system
2. **Security Launch**: Antivirus and firewalls start first (background)
3. **Health Check**: `health_monitor.py --boot` verifies system resources
4. **Safe Mode Check**: If health fails, enter safe_mode.py
5. **Main Launch**: `pandora_launcher.py` starts if healthy
6. **Core Principles**: Ethics framework loaded and displayed
7. **Orchestrator**: Full fabric orchestrator starts all subsystems
8. **Continuous Monitoring**: Health and security systems run continuously

### Security Layers
1. **Quantum Mirror Firewall**: Dynamic port randomization with recursive self-audit
2. **Antiviral Firewall**: Static UFW rules with ClamAV scanning
3. **Fluid Firewall**: Adaptive iptables with process sandboxing
4. **Filesystem Monitor**: Real-time inotify-based file scanning
5. **Safe Mode**: Ultimate fallback for degraded operation

### AI Components
1. **SubroutineAI**: Environment detection and optimization
2. **Self-Learning Agent**: Ethical learning with external effect checks
3. **Stoic Adviser**: Calm guidance and rational recommendations
4. **Calm Cache Optimizer**: Smooth request handling
5. **Quantum Virtual Processor**: Entropy generation for security

---

## Ethical Framework Integration

### How Ethics Are Enforced

1. **Import Check**: All modules import `ethics.core_principles`
2. **Action Validation**: `CorePrinciples.check_action(description)` before external effects
3. **Principle Display**: `CorePrinciples.exemplify()` shown at startup
4. **Learning Boundaries**: Self-learning agent distinguishes internal vs external effects
5. **Transparency**: All decisions logged with ethical reasoning
6. **Continuous Reflection**: System monitoring includes ethical audit

### Ethical Decision Flow
```
Proposed Action
    ↓
Is it external effect? (affects users/systems)
    ↓
YES → CorePrinciples.check_action()
    ↓
    ├─→ PASS: Execute and log
    └─→ FAIL: Block and log reasoning
    
NO → Execute (internal computation)
```

---

## Key Design Principles

### 1. Safety First
- Multiple security layers
- Health checks before full startup
- Safe mode fallback always available
- Automatic quarantine of threats

### 2. Ethical Operation
- Universal moral framework
- Transparency in all actions
- Harm prevention priority
- Continuous self-examination

### 3. Adaptive Intelligence
- Environment-aware configuration
- Self-learning within boundaries
- Quantum randomness for security
- Recursive self-improvement

### 4. Graceful Degradation
- Health verification at boot
- Safe mode for minimal operation
- Automatic rollback on merge
- Error handling with fallbacks

### 5. Stoic Philosophy
- Accept what cannot be controlled
- Calm under pressure
- Rational decision-making
- Balance and moderation

---

## Use Cases

### 1. Secure Bootable USB
Create a portable, secure AI system that can boot on any compatible hardware with automatic security and health verification.

### 2. Development Environment
Provide a complete AI development platform with built-in ethics, security, and monitoring.

### 3. Educational Platform
Demonstrate integration of philosophical principles, quantum concepts, and AI in a practical system.

### 4. Research Testbed
Experiment with quantum-inspired randomization, recursive self-audit, and ethical AI boundaries.

### 5. Secure AI Server
Deploy as a systemd service for always-on AI with multiple security layers and automatic health monitoring.

---

## File Statistics

### Total Files: 36 files
- Python scripts: 17
- Markdown documentation: 5
- Shell scripts: 7
- Configuration files: 3
- Batch files: 1
- Desktop entries: 1
- Service files: 1
- Archive files: 1

### Lines of Code (Approximate)
- Python: ~1,800 lines
- Bash: ~200 lines
- Markdown: ~800 lines
- Configuration: ~50 lines

---

## Important Notes for Gemini

### When Using This Documentation

1. **Context**: This is a philosophical AI system integrating ethics from multiple traditions
2. **Safety**: Security is paramount with multiple defensive layers
3. **Ethics**: All actions should be evaluated through CorePrinciples.check_action()
4. **Hardware**: Designed for Intel CPU + NVIDIA GPU systems
5. **Platform**: Primary target is Linux (Ubuntu/Debian), limited Windows support

### Key Concepts to Understand

1. **Fabric Architecture**: All components work as threads in a unified fabric
2. **Quantum Simulation**: Not true quantum computing, but quantum-inspired randomization
3. **Recursive Audit**: ML-over-ML concept where AI troubleshoots AI
4. **Ethical Boundaries**: Distinction between internal computation and external effects
5. **Stoic Operation**: Calm, rational, accepting of limitations

### Critical Files for System Understanding

1. **ethics/core_principles.py**: The moral foundation of all operations
2. **security/quantum_mirror_firewall.py**: Most complex security component
3. **startup/pandora_launcher.py**: Main entry point for full system
4. **self_learning_agent.py**: Demonstrates ethical learning implementation
5. **launch.sh**: Complete startup sequence with security and health checks

---

## Quick Reference Commands

### Installation
```bash
bash install_requirements.sh
```

### Manual Launch
```bash
bash launch.sh
```

### Health Check
```bash
python3 startup/health_monitor.py --boot
```

### Safe Mode
```bash
python3 startup/safe_mode.py
```

### Antivirus Scan
```bash
python3 security/antivirus_launcher.py /path/to/scan
```

### Fluid Firewall
```bash
python3 security/fluid_firewall.py --pid <aios_pid> --interval 90
```

---

## Diagnostic System

### diagnostic_system.py
**Purpose**: Comprehensive diagnostic and monitoring system for Pandora AIOS

**Philosophy**: Transparency, self-examination, continuous monitoring—know thyself

**Key Features**:
1. **Full System Diagnostics**: Complete health check of all components
2. **Hardware Detection**: CPU, Memory, Disk, GPU monitoring
3. **Dependency Verification**: Python packages and system packages
4. **Security Service Status**: ClamAV, UFW, firewall checks
5. **File Integrity Checks**: Verifies all Pandora files are present
6. **Health Log Analysis**: Scans logs for errors and issues
7. **Continuous Monitoring**: Real-time resource tracking
8. **JSON Report Generation**: Detailed reports for analysis

**Main Class**: `PandoraDiagnostics`

**Key Methods**:
- `check_system_info()`: Platform, Python version, architecture
- `check_hardware()`: CPU count/usage, memory, disk space
- `check_nvidia_gpu()`: GPU detection, driver version, memory, temperature
- `check_intel_cpu()`: Intel CPU detection and identification
- `check_dependencies()`: Python and system package verification
- `check_security_services()`: ClamAV and firewall status
- `check_pandora_files()`: Verifies all project files exist
- `check_health_logs()`: Scans logs for error keywords
- `run_full_diagnostic()`: Complete diagnostic suite
- `save_report()`: Export results to JSON
- `continuous_monitoring()`: Monitor resources over time

**Command-Line Usage**:
```bash
# Run full diagnostic
python3 diagnostic_system.py --full

# Run with custom output
python3 diagnostic_system.py --full --output /tmp/report.json

# Continuous monitoring (30s interval, 5min duration)
python3 diagnostic_system.py --monitor --interval 30 --duration 300

# Custom log location
python3 diagnostic_system.py --full --log /tmp/diagnostics.log
```

**Diagnostic Categories**:

1. **System Information**:
   - Operating system and version
   - Architecture (x86_64, ARM, etc.)
   - Hostname
   - Python version and executable path

2. **Hardware Resources**:
   - CPU core count and utilization
   - Total and available memory
   - Disk space usage
   - Real-time percentage metrics

3. **GPU Status**:
   - NVIDIA GPU availability
   - Driver and CUDA versions
   - GPU memory (total, used, free)
   - Temperature and utilization per GPU
   - Supports multiple GPUs

4. **CPU Verification**:
   - Intel CPU detection
   - CPU model name extraction
   - Cross-platform (Linux/Windows)

5. **Dependencies**:
   - Python: psutil, flask, flask_qrcode, pyclamd, inotify_simple
   - System: ufw, clamav, clamav-daemon
   - Lists missing dependencies

6. **Security Services**:
   - ClamAV daemon status (active/inactive)
   - UFW firewall status
   - Firewall rule verification

7. **File Integrity**:
   - Verifies all core Pandora files present
   - Checks root directory and extracted files
   - Reports missing files with paths

8. **Health Logs**:
   - Scans multiple log files
   - Detects error keywords: critical, fail, panic, error
   - Reports log status and findings

**Report Structure**:
```json
{
  "timestamp": "ISO-8601 datetime",
  "system_info": { ... },
  "hardware": { ... },
  "nvidia_gpu": { ... },
  "intel_cpu": { ... },
  "dependencies": { ... },
  "security_services": { ... },
  "pandora_files": { ... },
  "health_logs": { ... },
  "runtime_seconds": 0.15
}
```

**Health Status Levels**:
- **HEALTHY**: All checks passed, no issues
- **OPERATIONAL**: Some warnings but functional
- **DEGRADED**: Critical issues detected

**Critical Issues Detected**:
- NVIDIA GPU not available
- Missing critical dependencies
- Security services not running
- Core files missing

**Integration Points**:
- Can be called from startup scripts for pre-launch verification
- Continuous monitoring mode for production systems
- JSON reports for automated alerting
- Log integration with other Pandora systems

**Dependencies**: 
- Optional: psutil (for detailed hardware metrics)
- Standard library: platform, subprocess, json, os, sys, time

**Log Fallback**: If `/var/log/` is not writable, automatically falls back to `/tmp/` directory

**Use Cases**:
1. **Pre-Launch Verification**: Run before starting AIOS to ensure system readiness
2. **Troubleshooting**: Diagnose why system isn't working properly
3. **Continuous Monitoring**: Track resource usage during operation
4. **CI/CD Integration**: Automated testing and validation
5. **Documentation**: Generate system reports for support

**Example Integration in launch.sh**:
```bash
#!/bin/bash
# Run diagnostics before launch
python3 diagnostic_system.py --full --output /tmp/pre_launch_report.json

# Check exit code
if [ $? -ne 0 ]; then
    echo "Diagnostic failed! See /tmp/pre_launch_report.json"
    python3 startup/safe_mode.py
    exit 1
fi

# Continue with normal launch...
```

**Monitoring Output Example**:
```
[2025-11-14T23:20:03.523147][INFO] Platform: Linux 6.11.0-1018-azure
[2025-11-14T23:20:03.523147][INFO] Architecture: x86_64
[2025-11-14T23:20:03.523214][INFO] CPU: 4 cores, 15.2% used
[2025-11-14T23:20:03.523214][INFO] Memory: 6.8GB free / 8.0GB total (15% used)
[2025-11-14T23:20:03.523292][WARNING] NVIDIA GPU not detected
[2025-11-14T23:20:03.616100][INFO] System status: OPERATIONAL
```

---

## Conclusion

Pandora AIOS represents a comprehensive attempt to create an AI system that embodies universal ethical principles while maintaining robust security and adaptive intelligence. The system architecture emphasizes safety, transparency, and graceful degradation, with multiple layers of protection and fallback mechanisms.

The integration of philosophical and religious principles from diverse traditions creates a unique moral framework that guides all system operations. This is not merely theoretical but implemented through code-level checks and validations.

The security architecture employs innovative concepts like quantum-inspired randomization, recursive self-audit (ML-over-ML), and adaptive defense postures that continuously evolve.

All components work together as a unified "fabric" where each thread serves the whole, embodying the Stoic and Eastern philosophical principles of harmony, balance, and service without attachment to outcomes.

---

**This document contains complete information on all files in the Pandora AIOS project and can be used as a comprehensive prompt for Gemini or other AI systems to understand, modify, or extend the project.**
