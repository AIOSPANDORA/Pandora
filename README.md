# Pandora AIOS

An AI Computer Operating System - A lightweight, AI-powered operating system framework that demonstrates core OS concepts enhanced with AI capabilities.

## Overview

Pandora AIOS is an educational and experimental project that combines traditional operating system concepts with artificial intelligence. It provides a virtual environment where you can:

- Manage processes with AI-assisted scheduling
- Monitor system health with AI analysis
- Interact through a command-line shell
- Work with a virtual file system
- Experience AI-driven system optimization

## Documentation

- **[LOL:D Session Guide](./LOLD_SESSION_GUIDE.md)** - Comprehensive guide for conducting GitHub Copilot sessions with epistemic rigor, including LOL:D (Linguistically Optimized Ledger: Delta) and LOL:OB8 methodologies

## Features

### Core Components

1. **Kernel** - Manages processes, memory, and system resources
   - Process creation and termination
   - Memory management
   - Process state tracking
   - AI-assisted process handling

2. **AI Engine** - Provides intelligent system operations
   - Process priority optimization
   - Memory usage prediction
   - System health analysis
   - Action recommendations

3. **File System** - Virtual file system interface
   - File creation, reading, writing, and deletion
   - File listing and information
   - Storage management

4. **Shell** - Interactive command-line interface
   - Process management commands
   - File system operations
   - AI engine control
   - System monitoring

## Installation

```bash
# Clone the repository
git clone https://github.com/janschulzik-cmyk/Pandora.git
cd Pandora

# No external dependencies required - uses Python standard library
```

## Usage

### Running Pandora AIOS

```bash
python main.py
```

### Available Shell Commands

**Process Management:**
- `ps` - List all processes
- `create <name> [memory] [ai]` - Create a new process
- `kill <pid>` - Kill a process by PID

**System Monitoring:**
- `mem` - Show memory information with AI prediction
- `health` - Show AI-powered system health analysis

**AI Engine:**
- `ai status` - Check AI engine status
- `ai stats` - View AI engine statistics
- `ai enable` - Enable AI engine
- `ai disable` - Disable AI engine

**File System:**
- `ls` - List files
- `touch <file>` - Create a new file
- `cat <file>` - Display file content
- `echo <text> > <file>` - Write text to file
- `rm <file>` - Remove a file

**General:**
- `help` - Show all available commands
- `exit` or `shutdown` - Exit the shell

## Example Session

```
pandora> ps
PID        Name                 State        Memory     AI   
-----------------------------------------------------------------
a1b2c3d4   init                 ready        5           
e5f6g7h8   ai-daemon            ready        20         ✓

pandora> create myprocess 30 ai
Process created: i9j0k1l2 (myprocess)

pandora> mem
Memory Information:
  Total:  1024MB
  Used:   55MB
  Free:   969MB
  Usage:  5.4%
  
  AI Predicted Usage: 55MB

pandora> health
System Health Analysis:
  Status: HEALTHY
  Score:  100/100

pandora> touch data.txt

pandora> echo Hello from Pandora AIOS > data.txt
Written to data.txt

pandora> cat data.txt
Hello from Pandora AIOS

pandora> ls
Name                           Size       Modified
------------------------------------------------------------
data.txt                       24         2025-11-14 04:12
```

## Architecture

```
┌─────────────────────────────────────────┐
│              Shell (CLI)                │
│    User interaction & commands          │
└─────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
┌───────▼──────┐ ┌──▼───────┐ ┌▼────────────┐
│   Kernel     │ │AI Engine │ │ File System │
│ - Processes  │ │- Analysis│ │- Files      │
│ - Memory     │ │- Predict │ │- Storage    │
│ - Scheduler  │ │- Optimize│ │             │
└──────────────┘ └──────────┘ └─────────────┘
```

## Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific test module
python -m unittest tests.test_kernel
python -m unittest tests.test_ai_engine
python -m unittest tests.test_filesystem
```

## Development

### Project Structure

```
Pandora/
├── pandora_aios/
│   ├── __init__.py       # Package initialization
│   ├── kernel.py         # Core kernel functionality
│   ├── ai_engine.py      # AI intelligence module
│   ├── filesystem.py     # Virtual file system
│   └── shell.py          # Command-line interface
├── tests/
│   ├── test_kernel.py
│   ├── test_ai_engine.py
│   └── test_filesystem.py
├── main.py               # Entry point
├── setup.py              # Package setup
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## AI Capabilities

The AI Engine provides several intelligent features:

1. **Process Priority Optimization** - Automatically adjusts process priorities based on AI assistance flags and system state

2. **Memory Usage Prediction** - Predicts future memory requirements based on current usage patterns and process count

3. **System Health Analysis** - Continuously monitors system health and provides:
   - Health scores (0-100)
   - Issue identification
   - Actionable recommendations

4. **Adaptive Recommendations** - Suggests actions based on system state:
   - Memory management advice
   - Process optimization tips
   - Resource allocation suggestions

## Technical Details

- **Language**: Python 3.7+
- **Dependencies**: None (uses Python standard library only)
- **Architecture**: Modular, object-oriented design
- **Testing**: Unit tests with unittest framework
- **License**: MIT

## Future Enhancements

Potential improvements for future versions:

- [ ] Advanced scheduling algorithms (Round-robin, Priority-based)
- [ ] Persistent file system with actual disk storage
- [ ] Network stack simulation
- [ ] Machine learning integration for better predictions
- [ ] Multi-threading support
- [ ] Inter-process communication (IPC)
- [ ] Device driver abstraction
- [ ] GUI interface

## Contributing

Contributions are welcome! This is an educational project designed to help understand OS concepts and AI integration.

## License

MIT License - See LICENSE file for details

## Acknowledgments

Created as a demonstration of AI-powered operating system concepts, combining traditional OS design with modern artificial intelligence capabilities.

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
