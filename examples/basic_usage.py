#!/usr/bin/env python3
"""
Example: Basic Usage of Pandora AIOS API

This example demonstrates how to use Pandora AIOS programmatically
without the interactive shell.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pandora_aios import Kernel, AIEngine, FileSystem


def main():
    """Demonstrate basic API usage"""
    
    print("=== Pandora AIOS API Example ===\n")
    
    # Initialize components
    kernel = Kernel()
    ai_engine = AIEngine()
    filesystem = FileSystem()
    
    # Boot the system
    print("1. Booting kernel...")
    kernel.boot()
    print()
    
    # Create processes
    print("2. Creating processes...")
    p1 = kernel.create_process("web-server", memory=50, priority=5)
    p2 = kernel.create_process("database", memory=100, priority=8)
    p3 = kernel.create_process("ai-worker", memory=75, ai_assisted=True)
    
    print(f"   Created: {p1.name} (PID: {p1.pid})")
    print(f"   Created: {p2.name} (PID: {p2.pid})")
    print(f"   Created: {p3.name} (PID: {p3.pid}, AI-assisted)")
    print()
    
    # Check memory
    print("3. Memory status:")
    mem_info = kernel.get_memory_info()
    print(f"   Used: {mem_info['used']}MB / {mem_info['total']}MB")
    print(f"   Free: {mem_info['free']}MB")
    print()
    
    # AI optimization
    print("4. AI Process Priority Optimization:")
    processes = kernel.list_processes()
    priorities = ai_engine.optimize_process_priority(processes)
    for pid, priority in priorities.items():
        proc = kernel.get_process(pid)
        print(f"   {proc.name}: Priority {priority}")
    print()
    
    # AI health analysis
    print("5. AI System Health Analysis:")
    health = ai_engine.analyze_system_health(processes, mem_info)
    print(f"   Status: {health['status']}")
    print(f"   Health Score: {health['score']}/100")
    if health['recommendations']:
        print("   Recommendations:")
        for rec in health['recommendations']:
            print(f"     - {rec}")
    print()
    
    # File operations
    print("6. File system operations:")
    filesystem.create_file("/config.txt", "server_port=8080\ndb_host=localhost")
    filesystem.create_file("/data.log", "System started successfully")
    
    files = filesystem.list_files()
    print(f"   Created {len(files)} files:")
    for file in files:
        print(f"     - {file.name} ({file.size} bytes)")
    print()
    
    # Read file
    print("7. Reading configuration:")
    config = filesystem.read_file("/config.txt")
    print(f"   {config}")
    print()
    
    # AI prediction
    print("8. AI Memory Prediction:")
    predicted = ai_engine.predict_memory_usage(mem_info['used'], len(processes))
    print(f"   Current: {mem_info['used']}MB")
    print(f"   Predicted: {predicted}MB")
    print()
    
    # AI statistics
    print("9. AI Engine Statistics:")
    stats = ai_engine.get_stats()
    print(f"   Enabled: {stats['enabled']}")
    print(f"   Tasks Processed: {stats['tasks_processed']}")
    print()
    
    # Cleanup
    print("10. Shutting down...")
    kernel.shutdown()
    print("Done!")


if __name__ == "__main__":
    main()
