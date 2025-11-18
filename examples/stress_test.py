#!/usr/bin/env python3
"""
Example: Stress Testing Pandora AIOS

This example creates many processes to test system limits
and AI recommendations under load.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pandora_aios import Kernel, AIEngine
import random


def main():
    """Stress test the system"""
    
    print("=== Pandora AIOS Stress Test ===\n")
    
    kernel = Kernel()
    ai_engine = AIEngine()
    kernel.boot()
    
    # Create many processes
    print("Creating 15 processes...")
    for i in range(15):
        name = f"process-{i}"
        memory = random.randint(10, 50)
        ai_assisted = (i % 3 == 0)  # Every 3rd process is AI-assisted
        
        try:
            p = kernel.create_process(name, memory=memory, ai_assisted=ai_assisted)
            ai_marker = "🤖" if ai_assisted else "  "
            print(f"  {ai_marker} {p.name} created (PID: {p.pid}, {memory}MB)")
        except MemoryError as e:
            print(f"  ❌ Failed to create {name}: {e}")
            break
    
    print()
    
    # Check system state
    processes = kernel.list_processes()
    mem_info = kernel.get_memory_info()
    
    print(f"System State:")
    print(f"  Processes: {len(processes)}")
    print(f"  Memory Used: {mem_info['used']}MB / {mem_info['total']}MB")
    print(f"  Memory Free: {mem_info['free']}MB")
    print(f"  Usage: {(mem_info['used']/mem_info['total']*100):.1f}%")
    print()
    
    # AI Analysis under stress
    print("AI Analysis:")
    health = ai_engine.analyze_system_health(processes, mem_info)
    print(f"  Status: {health['status'].upper()}")
    print(f"  Health Score: {health['score']}/100")
    
    if health['issues']:
        print(f"\n  Issues Detected:")
        for issue in health['issues']:
            print(f"    ⚠️  {issue}")
    
    if health['recommendations']:
        print(f"\n  AI Recommendations:")
        for rec in health['recommendations']:
            print(f"    💡 {rec}")
    
    print()
    
    # Get AI recommendation
    system_state = {"memory": mem_info}
    action = ai_engine.recommend_action(system_state)
    print(f"AI Recommendation: {action}")
    print()
    
    # Optimize priorities
    print("Optimizing process priorities...")
    priorities = ai_engine.optimize_process_priority(processes)
    ai_procs = [p for p in processes if p.ai_assisted]
    regular_procs = [p for p in processes if not p.ai_assisted]
    
    print(f"  AI-assisted processes ({len(ai_procs)}): Higher priority")
    print(f"  Regular processes ({len(regular_procs)}): Standard priority")
    print()
    
    # Cleanup - kill some processes
    print("Cleaning up - terminating processes...")
    killed = 0
    for proc in processes[:5]:  # Kill first 5
        if kernel.kill_process(proc.pid):
            print(f"  ✓ Terminated {proc.name}")
            killed += 1
    
    print(f"\nTerminated {killed} processes")
    
    # Final state
    mem_info_after = kernel.get_memory_info()
    print(f"Memory freed: {mem_info_after['free'] - mem_info['free']}MB")
    print()
    
    kernel.shutdown()
    print("Stress test completed!")


if __name__ == "__main__":
    main()
