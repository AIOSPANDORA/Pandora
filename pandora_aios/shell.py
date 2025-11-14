"""
Shell - Interactive command-line interface
Provides user interaction with the AI OS
"""

import sys
from typing import Optional
from .kernel import Kernel
from .ai_engine import AIEngine
from .filesystem import FileSystem


class Shell:
    """Command-line shell for Pandora AIOS"""
    
    def __init__(self, kernel: Kernel, ai_engine: AIEngine, filesystem: FileSystem):
        self.kernel = kernel
        self.ai_engine = ai_engine
        self.filesystem = filesystem
        self.running = False
        self.commands = {
            "help": self.cmd_help,
            "ps": self.cmd_ps,
            "create": self.cmd_create_process,
            "kill": self.cmd_kill_process,
            "mem": self.cmd_memory,
            "health": self.cmd_health,
            "ai": self.cmd_ai,
            "ls": self.cmd_ls,
            "touch": self.cmd_touch,
            "cat": self.cmd_cat,
            "echo": self.cmd_echo,
            "rm": self.cmd_rm,
            "exit": self.cmd_exit,
            "shutdown": self.cmd_exit,
        }
        
    def start(self):
        """Start the shell"""
        self.running = True
        print("\nWelcome to Pandora AIOS Shell v0.1.0")
        print("Type 'help' for available commands\n")
        
        while self.running:
            try:
                command = input("pandora> ").strip()
                if command:
                    self.execute(command)
            except KeyboardInterrupt:
                print("\nUse 'exit' or 'shutdown' to quit")
            except EOFError:
                break
                
    def execute(self, command_line: str):
        """Execute a command"""
        parts = command_line.split()
        if not parts:
            return
            
        cmd = parts[0].lower()
        args = parts[1:]
        
        if cmd in self.commands:
            try:
                self.commands[cmd](args)
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for available commands")
            
    def cmd_help(self, args):
        """Show help information"""
        print("\nAvailable Commands:")
        print("  help              - Show this help message")
        print("  ps                - List all processes")
        print("  create <name>     - Create a new process")
        print("  kill <pid>        - Kill a process by PID")
        print("  mem               - Show memory information")
        print("  health            - Show system health analysis")
        print("  ai <status|stats> - AI engine control and stats")
        print("  ls                - List files")
        print("  touch <file>      - Create a new file")
        print("  cat <file>        - Display file content")
        print("  echo <text> > <f> - Write text to file")
        print("  rm <file>         - Remove a file")
        print("  exit/shutdown     - Exit the shell")
        print()
        
    def cmd_ps(self, args):
        """List processes"""
        processes = self.kernel.list_processes()
        if not processes:
            print("No processes running")
            return
            
        print(f"\n{'PID':<10} {'Name':<20} {'State':<12} {'Memory':<10} {'AI':<5}")
        print("-" * 65)
        for proc in processes:
            ai_marker = "✓" if proc.ai_assisted else " "
            print(f"{proc.pid:<10} {proc.name:<20} {proc.state.value:<12} {proc.memory:<10} {ai_marker:<5}")
        print()
        
    def cmd_create_process(self, args):
        """Create a new process"""
        if not args:
            print("Usage: create <name> [memory] [ai]")
            return
            
        name = args[0]
        memory = int(args[1]) if len(args) > 1 else 10
        ai_assisted = len(args) > 2 and args[2].lower() == "ai"
        
        try:
            process = self.kernel.create_process(name, memory, ai_assisted=ai_assisted)
            print(f"Process created: {process.pid} ({name})")
        except MemoryError as e:
            print(f"Failed to create process: {e}")
            
    def cmd_kill_process(self, args):
        """Kill a process"""
        if not args:
            print("Usage: kill <pid>")
            return
            
        pid = args[0]
        if self.kernel.kill_process(pid):
            print(f"Process {pid} terminated")
        else:
            print(f"Process {pid} not found")
            
    def cmd_memory(self, args):
        """Show memory information"""
        mem_info = self.kernel.get_memory_info()
        print(f"\nMemory Information:")
        print(f"  Total:  {mem_info['total']}MB")
        print(f"  Used:   {mem_info['used']}MB")
        print(f"  Free:   {mem_info['free']}MB")
        print(f"  Usage:  {(mem_info['used']/mem_info['total']*100):.1f}%")
        
        # AI prediction
        processes = self.kernel.list_processes()
        predicted = self.ai_engine.predict_memory_usage(mem_info['used'], len(processes))
        print(f"\n  AI Predicted Usage: {predicted}MB")
        print()
        
    def cmd_health(self, args):
        """Show system health analysis"""
        processes = self.kernel.list_processes()
        mem_info = self.kernel.get_memory_info()
        health = self.ai_engine.analyze_system_health(processes, mem_info)
        
        print(f"\nSystem Health Analysis:")
        print(f"  Status: {health['status'].upper()}")
        print(f"  Score:  {health['score']}/100")
        
        if health['issues']:
            print(f"\n  Issues:")
            for issue in health['issues']:
                print(f"    - {issue}")
                
        if health['recommendations']:
            print(f"\n  Recommendations:")
            for rec in health['recommendations']:
                print(f"    - {rec}")
        print()
        
    def cmd_ai(self, args):
        """AI engine control"""
        if not args:
            print("Usage: ai <status|stats|enable|disable>")
            return
            
        subcmd = args[0].lower()
        
        if subcmd == "status":
            status = "Enabled" if self.ai_engine.enabled else "Disabled"
            print(f"AI Engine: {status}")
        elif subcmd == "stats":
            stats = self.ai_engine.get_stats()
            print(f"\nAI Engine Statistics:")
            print(f"  Enabled:         {stats['enabled']}")
            print(f"  Learning Mode:   {stats['learning_mode']}")
            print(f"  Tasks Processed: {stats['tasks_processed']}")
            print()
        elif subcmd == "enable":
            self.ai_engine.enable()
            print("AI Engine enabled")
        elif subcmd == "disable":
            self.ai_engine.disable()
            print("AI Engine disabled")
        else:
            print(f"Unknown AI command: {subcmd}")
            
    def cmd_ls(self, args):
        """List files"""
        files = self.filesystem.list_files()
        if not files:
            print("No files")
            return
            
        print(f"\n{'Name':<30} {'Size':<10} {'Modified'}")
        print("-" * 60)
        for file in files:
            import datetime
            mod_time = datetime.datetime.fromtimestamp(file.modified_at).strftime('%Y-%m-%d %H:%M')
            print(f"{file.name:<30} {file.size:<10} {mod_time}")
        print()
        
    def cmd_touch(self, args):
        """Create a file"""
        if not args:
            print("Usage: touch <filename>")
            return
            
        try:
            file = self.filesystem.create_file(args[0])
            print(f"File created: {file.name}")
        except FileExistsError as e:
            print(f"Error: {e}")
            
    def cmd_cat(self, args):
        """Display file content"""
        if not args:
            print("Usage: cat <filename>")
            return
            
        try:
            content = self.filesystem.read_file(args[0])
            print(content if content else "(empty file)")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            
    def cmd_echo(self, args):
        """Write to file"""
        if len(args) < 3 or args[-2] != '>':
            print("Usage: echo <text> > <filename>")
            return
            
        text = ' '.join(args[:-2])
        filename = args[-1]
        
        try:
            # Try to write to existing file or create new one
            if filename in [f.name for f in self.filesystem.list_files()]:
                self.filesystem.write_file(filename, text)
            else:
                self.filesystem.create_file(filename, text)
            print(f"Written to {filename}")
        except Exception as e:
            print(f"Error: {e}")
            
    def cmd_rm(self, args):
        """Remove a file"""
        if not args:
            print("Usage: rm <filename>")
            return
            
        if self.filesystem.delete_file(args[0]):
            print(f"File removed: {args[0]}")
        else:
            print(f"File not found: {args[0]}")
            
    def cmd_exit(self, args):
        """Exit the shell"""
        self.running = False
        print("Goodbye!")
