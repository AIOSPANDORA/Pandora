# AIOS Pandora (Cubic USB/Live Boot Ready)

## Overview

Pandora AIOS is an AI Operating System built with stoic philosophy at its core. It provides intelligent system management, security monitoring, and adaptive resource optimization.

## Features

- **Intelligent System Monitoring**: Real-time CPU, memory, and disk monitoring
- **Stoic AI Adviser**: Context-aware wisdom and guidance
- **Quantum Virtual Processor**: Simulated quantum computing capabilities
- **Calm Cache Optimizer**: LRU cache with expiration for smooth performance
- **Security Framework**: Antiviral firewall and fluid security layers
- **Assimilation Portal**: Web-based interface for device integration
- **GUI Control Panel**: Desktop application for system management

## Quick Start

### Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure System**
   ```bash
   # Edit pandora_config.py to customize settings
   python3 pandora_config.py  # Validate configuration
   ```

3. **Run Individual Components**
   ```bash
   # Launch GUI
   python3 pandora_gui.py
   
   # Start Assimilation Portal
   python3 assimilate.py
   
   # Monitor System Health
   python3 system_monitoring.py
   ```

## Setup for USB/Live Boot
## Setup

1. **Integrate into ISO**  
   Use [Cubic](https://launchpad.net/cubic) to open your base Ubuntu ISO.  
   Copy the `aios-pandora-usb/` folder to `/` of your live ISO environment during Cubic setup.

2. **Set Permissions**  
   Run `chmod +x /aios-pandora-usb/*.sh /aios-pandora-usb/startup/*.py`.

3. **Autostart**  
   Copy `aios-autostart.desktop` to `/etc/skel/.config/autostart/` during Cubic.

4. **(Optional) Headless/Daemon Mode**  
   Copy the sample systemd unit into `/etc/systemd/system/`, then  
   `systemctl enable pandora-aios` in Cubic.

5. **Burn ISO to USB & Boot!**  
   Use Etcher or Rufus to create your bootable USB from the finished ISO.

## On Boot

- Antivirus and firewalls start before AIOS.
- Pandora always runs safe by default; full orchestrator only starts if health checks pass.
- Pandora always runs safe by default; full orchestrator only starts if health checks pass.

## Components

### Core Modules

- **SubroutineAI**: Environment detection and optimization recommendations
- **CalmCacheOptimizer**: Intelligent caching with LRU and expiration
- **QuantumVirtualProcessor**: Quantum gate simulation (H, X, Z, T gates)
- **SystemMonitoring**: Health checks and resource monitoring
- **StoicAdviser**: Context-aware philosophical guidance

### Security

- **AntiviralFirewall**: UFW-based firewall with ClamAV integration
- **FluidFirewall**: Adaptive security layer
- **QuantumMirrorFirewall**: Advanced threat detection

### Utilities

- **Assimilation Portal**: Web interface for device pairing (Flask-based)
- **GUI Control Panel**: Tkinter-based desktop control
- **Health Monitor**: Startup health verification
- **Safe Mode**: Fallback mode for system errors

## Configuration

All settings are centralized in `pandora_config.py`:

- Cache size and expiration
- Monitoring thresholds
- Security settings
- Port configuration
- AI model parameters
- Ethics framework settings

## Ethics Framework

Pandora AIOS includes a comprehensive ethics framework based on:
- Transparency and openness
- User privacy and consent
- Harm prevention
- Stoic virtues (wisdom, courage, justice, temperance)

See `ETHICS.md` and `ethics/` directory for full details.

## Hardware Requirements

- **Minimum**: 2GB RAM, 1 CPU core
- **Recommended**: 4GB+ RAM, 2+ CPU cores
- **Optional**: NVIDIA GPU for advanced AI features

## Philosophy

Pandora AIOS embodies stoic principles:
- **Accept what you cannot control**: Graceful degradation under constraints
- **Optimize what you can**: Intelligent resource management
- **Calm under pressure**: No panic, only rational responses
- **Virtue in action**: Ethical operation at all times

## Support

For issues and questions, refer to the ethics documentation or examine the code - all modules are designed to be self-documenting and transparent.

## License

Open source - designed for accessibility and community improvement.
- Pandora always runs safe by default; full orchestrator only starts if health checks pass.
