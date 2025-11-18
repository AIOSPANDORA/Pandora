"""
Kernel Module - Core OS functionality
Manages processes, memory, and system resources
"""

import time
import uuid
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class ProcessState(Enum):
    """Process states"""
    NEW = "new"
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    TERMINATED = "terminated"


@dataclass
class Process:
    """Represents a process in the system"""
    pid: str
    name: str
    state: ProcessState = ProcessState.NEW
    priority: int = 0
    memory: int = 0
    created_at: float = field(default_factory=time.time)
    ai_assisted: bool = False
    
    def __post_init__(self):
        if not self.pid:
            self.pid = str(uuid.uuid4())[:8]


class Kernel:
    """Core kernel that manages system resources"""
    
    def __init__(self):
        self.processes: Dict[str, Process] = {}
        self.memory_total = 1024  # MB
        self.memory_used = 0
        self.running = False
        
    def boot(self):
        """Boot the kernel"""
        self.running = True
        print("Pandora AIOS Kernel v0.1.0 booting...")
        print(f"Total Memory: {self.memory_total}MB")
        
    def shutdown(self):
        """Shutdown the kernel"""
        print("Shutting down Pandora AIOS...")
        for pid in list(self.processes.keys()):
            self.kill_process(pid)
        self.running = False
        
    def create_process(self, name: str, memory: int = 10, 
                      priority: int = 0, ai_assisted: bool = False) -> Process:
        """Create a new process"""
        if self.memory_used + memory > self.memory_total:
            raise MemoryError("Not enough memory to create process")
            
        process = Process(
            pid=str(uuid.uuid4())[:8],
            name=name,
            memory=memory,
            priority=priority,
            ai_assisted=ai_assisted,
            state=ProcessState.READY
        )
        
        self.processes[process.pid] = process
        self.memory_used += memory
        return process
        
    def kill_process(self, pid: str) -> bool:
        """Terminate a process"""
        if pid not in self.processes:
            return False
            
        process = self.processes[pid]
        process.state = ProcessState.TERMINATED
        self.memory_used -= process.memory
        del self.processes[pid]
        return True
        
    def list_processes(self) -> List[Process]:
        """List all processes"""
        return list(self.processes.values())
        
    def get_process(self, pid: str) -> Optional[Process]:
        """Get a process by PID"""
        return self.processes.get(pid)
        
    def get_memory_info(self) -> Dict[str, int]:
        """Get memory information"""
        return {
            "total": self.memory_total,
            "used": self.memory_used,
            "free": self.memory_total - self.memory_used
        }
