# Pandora AIOS Architecture Guide

## System Overview

Pandora AIOS (AI Operating System) is a lightweight, educational operating system framework that demonstrates how artificial intelligence can be integrated into core OS operations. The system is built with a modular architecture using Python.

## Core Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     User Interface Layer                    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Shell (Command-Line Interface)          │  │
│  │  - Command parsing and execution                     │  │
│  │  - User interaction and feedback                     │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
                            │
                            │ Commands
                            ▼
┌────────────────────────────────────────────────────────────┐
│                     System Services Layer                   │
│                                                              │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────┐ │
│  │   Kernel     │  │  AI Engine    │  │  File System    │ │
│  │              │  │               │  │                 │ │
│  │ - Processes  │  │ - Optimize    │  │ - Files         │ │
│  │ - Memory     │  │ - Predict     │  │ - CRUD Ops      │ │
│  │ - Scheduler  │  │ - Analyze     │  │ - Storage       │ │
│  │              │  │ - Recommend   │  │                 │ │
│  └──────────────┘  └───────────────┘  └─────────────────┘ │
│         ▲                  ▲                    ▲          │
│         │                  │                    │          │
│         └──────────────────┴────────────────────┘          │
│                    Shared State                            │
└────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Kernel Module (`kernel.py`)

The Kernel is the core of Pandora AIOS, responsible for:

**Process Management:**
- Process lifecycle (creation, execution, termination)
- Process state tracking (NEW, READY, RUNNING, WAITING, TERMINATED)
- Process identification via unique PIDs
- Support for AI-assisted processes

**Memory Management:**
- Virtual memory allocation (1024MB by default)
- Memory tracking per process
- Out-of-memory protection
- Memory usage reporting

**Key Classes:**
- `ProcessState` (Enum): Defines process states
- `Process` (Dataclass): Represents a system process
- `Kernel` (Class): Main kernel controller

**API Methods:**
```python
kernel.boot()                          # Initialize kernel
kernel.create_process(name, memory)    # Create new process
kernel.kill_process(pid)               # Terminate process
kernel.list_processes()                # Get all processes
kernel.get_memory_info()               # Memory statistics
kernel.shutdown()                      # Cleanup and shutdown
```

### 2. AI Engine (`ai_engine.py`)

The AI Engine provides intelligent system operations:

**Capabilities:**
1. **Process Priority Optimization**: Automatically adjusts priorities
2. **Memory Usage Prediction**: Forecasts memory requirements
3. **System Health Analysis**: Evaluates system state
4. **Action Recommendations**: Suggests optimizations

**AI Algorithms:**
- Priority optimization: AI-assisted processes get higher priority
- Memory prediction: Based on process count and current usage
- Health scoring: 0-100 scale based on memory and process metrics
- Adaptive recommendations: Context-aware suggestions

**Key Methods:**
```python
ai_engine.optimize_process_priority(processes)
ai_engine.predict_memory_usage(current, count)
ai_engine.analyze_system_health(processes, memory)
ai_engine.recommend_action(system_state)
```

### 3. File System (`filesystem.py`)

Virtual file system providing standard file operations:

**Features:**
- File creation, reading, writing, deletion
- File metadata (size, timestamps)
- Simple flat namespace (no directories yet)
- In-memory storage

**Key Classes:**
- `File` (Dataclass): Represents a file with metadata
- `FileSystem` (Class): File system controller

**API Methods:**
```python
fs.create_file(path, content)   # Create new file
fs.read_file(path)               # Read file content
fs.write_file(path, content)     # Update file
fs.delete_file(path)             # Remove file
fs.list_files()                  # List all files
```

### 4. Shell Interface (`shell.py`)

Interactive command-line interface:

**Command Categories:**

1. **Process Management**: ps, create, kill
2. **System Monitoring**: mem, health
3. **AI Control**: ai status, ai stats, ai enable/disable
4. **File Operations**: ls, touch, cat, echo, rm
5. **General**: help, exit, shutdown

**Shell Architecture:**
```python
Shell(kernel, ai_engine, filesystem)
  │
  ├─ Command Parser
  ├─ Command Router
  ├─ Command Handlers (13 commands)
  └─ Output Formatter
```

## Data Flow Examples

### Example 1: Creating a Process

```
User Input: "create web-server 50 ai"
    │
    ▼
Shell.execute("create web-server 50 ai")
    │
    ▼
Shell.cmd_create_process(["web-server", "50", "ai"])
    │
    ▼
Kernel.create_process(name="web-server", memory=50, ai_assisted=True)
    │
    ├─ Check memory availability
    ├─ Create Process object with unique PID
    ├─ Update memory tracking
    └─ Store in processes dictionary
    │
    ▼
Return: Process(pid="abc123", name="web-server", ...)
    │
    ▼
Shell: Display "Process created: abc123 (web-server)"
```

### Example 2: System Health Check

```
User Input: "health"
    │
    ▼
Shell.cmd_health()
    │
    ├─ Get processes from Kernel
    ├─ Get memory info from Kernel
    │
    ▼
AIEngine.analyze_system_health(processes, memory)
    │
    ├─ Calculate memory usage percentage
    ├─ Count processes
    ├─ Determine health score (0-100)
    ├─ Identify issues
    └─ Generate recommendations
    │
    ▼
Return: {status, score, issues, recommendations}
    │
    ▼
Shell: Format and display health report
```

## Design Principles

### 1. Modularity
Each component is independent and can be used separately:
```python
kernel = Kernel()
ai_engine = AIEngine()
filesystem = FileSystem()
```

### 2. Extensibility
Easy to add new features:
- New shell commands via `self.commands` dictionary
- New AI algorithms by extending AIEngine methods
- New process attributes by updating Process dataclass

### 3. Simplicity
- No external dependencies
- Pure Python standard library
- Clean, readable code
- Comprehensive docstrings

### 4. Educational Value
- Clear separation of concerns
- Well-documented architecture
- Example-driven learning
- Progressive complexity

## Testing Architecture

```
tests/
  ├── test_kernel.py       (7 tests)
  │   ├── Process creation/termination
  │   ├── Memory management
  │   └── State tracking
  │
  ├── test_ai_engine.py    (8 tests)
  │   ├── Optimization algorithms
  │   ├── Prediction accuracy
  │   └── Health analysis
  │
  └── test_filesystem.py   (7 tests)
      ├── File CRUD operations
      ├── Error handling
      └── Metadata tracking
```

## Performance Characteristics

- **Memory Overhead**: Minimal (~5MB for kernel)
- **Process Creation**: O(1) constant time
- **Process Lookup**: O(1) dictionary access
- **File Operations**: O(1) for single file ops
- **AI Analysis**: O(n) where n = number of processes

## Security Considerations

1. **Memory Protection**: Out-of-memory guards
2. **Resource Limits**: Configurable memory caps
3. **Input Validation**: Command argument checking
4. **Error Handling**: Graceful failure modes
5. **No Privilege Escalation**: Flat security model

## Future Architecture Enhancements

### Planned Features
1. **Process Scheduler**: Round-robin, priority-based algorithms
2. **IPC**: Inter-process communication mechanisms
3. **Directory Support**: Hierarchical file system
4. **Persistent Storage**: Disk-backed file system
5. **Network Stack**: Virtual networking simulation
6. **Multi-threading**: Concurrent process execution
7. **Advanced AI**: Machine learning models for predictions

### Architecture Evolution
```
Current: Monolithic kernel with AI advisory
    │
    ▼
Phase 2: Microkernel with AI services
    │
    ▼
Phase 3: Distributed AI-OS with ML integration
```

## Development Guidelines

### Adding a New Shell Command
```python
# 1. Add command to Shell.__init__
self.commands["newcmd"] = self.cmd_newcmd

# 2. Implement command handler
def cmd_newcmd(self, args):
    """Command description"""
    # Validation
    if not args:
        print("Usage: newcmd <arg>")
        return
    # Logic
    result = self.kernel.some_method(args[0])
    # Output
    print(f"Result: {result}")
```

### Adding AI Capability
```python
# In ai_engine.py
def new_ai_feature(self, input_data):
    """New AI capability description"""
    if not self.enabled:
        return None
    
    # AI logic here
    result = self._analyze(input_data)
    
    self.tasks_processed += 1
    return result
```

## Configuration

System parameters can be adjusted:
```python
# Memory size
kernel.memory_total = 2048  # 2GB

# AI settings
ai_engine.learning_mode = False
ai_engine.enabled = True
```

## Conclusion

Pandora AIOS demonstrates a clean, modular architecture for integrating AI into operating system operations. The system is designed for education, experimentation, and understanding how AI can enhance traditional OS functions.
