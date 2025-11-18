"""
File System - Virtual file system interface
Manages files and directories in the AI OS
"""

import os
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class File:
    """Represents a file in the virtual file system"""
    name: str
    content: str = ""
    size: int = 0
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        self.size = len(self.content)


class FileSystem:
    """Virtual file system"""
    
    def __init__(self):
        self.root: Dict[str, File] = {}
        self.current_dir = "/"
        
    def create_file(self, path: str, content: str = "") -> File:
        """Create a new file"""
        if path in self.root:
            raise FileExistsError(f"File {path} already exists")
            
        file = File(name=path, content=content)
        self.root[path] = file
        return file
        
    def read_file(self, path: str) -> str:
        """Read file content"""
        if path not in self.root:
            raise FileNotFoundError(f"File {path} not found")
        return self.root[path].content
        
    def write_file(self, path: str, content: str) -> None:
        """Write content to file"""
        if path not in self.root:
            raise FileNotFoundError(f"File {path} not found")
            
        file = self.root[path]
        file.content = content
        file.size = len(content)
        file.modified_at = time.time()
        
    def delete_file(self, path: str) -> bool:
        """Delete a file"""
        if path not in self.root:
            return False
        del self.root[path]
        return True
        
    def list_files(self) -> List[File]:
        """List all files"""
        return list(self.root.values())
        
    def get_file_info(self, path: str) -> Optional[File]:
        """Get file information"""
        return self.root.get(path)
        
    def get_total_size(self) -> int:
        """Get total size of all files"""
        return sum(file.size for file in self.root.values())
