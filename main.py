#!/usr/bin/env python3
"""
Pandora AIOS - Main Entry Point
Run the AI Operating System
"""

from pandora_aios import Kernel, AIEngine, Shell, FileSystem


def main():
    """Main entry point for Pandora AIOS"""
    print("=" * 60)
    print("  PANDORA AIOS - AI Computer Operating System")
    print("  Version 0.1.0")
    print("=" * 60)
    
    # Initialize components
    kernel = Kernel()
    ai_engine = AIEngine()
    filesystem = FileSystem()
    
    # Boot the kernel
    kernel.boot()
    
    # Create some initial system processes
    kernel.create_process("init", memory=5, priority=10)
    kernel.create_process("ai-daemon", memory=20, priority=8, ai_assisted=True)
    
    print(f"AI Engine: {'Enabled' if ai_engine.enabled else 'Disabled'}")
    print(f"Learning Mode: {'Active' if ai_engine.learning_mode else 'Inactive'}")
    print()
    
    # Start the shell
    shell = Shell(kernel, ai_engine, filesystem)
    
    try:
        shell.start()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        # Shutdown
        kernel.shutdown()
        print("System halted.")


if __name__ == "__main__":
    main()
