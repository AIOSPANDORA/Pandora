# Pandora AIOS

An AI Computer Operating System - A lightweight, AI-powered operating system framework that demonstrates core OS concepts enhanced with AI capabilities.

## Overview

Pandora AIOS is an educational and experimental project that combines traditional operating system concepts with artificial intelligence. It provides a virtual environment where you can:

- Manage processes with AI-assisted scheduling
- Monitor system health with AI analysis
- Interact through a command-line shell
- Work with a virtual file system
- Experience AI-driven system optimization

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

