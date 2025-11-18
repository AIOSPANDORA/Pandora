#!/usr/bin/env python3
"""
Pandora Quantum System Launcher
================================

Universal cross-platform launcher for Pandora's quantum interface with Hamiltonian support.
This launcher provides:
- Hardware detection (NVIDIA GPU, Intel CPU, ASUS ROG)
- Quantum profile selection (Alternative, Castle, Hive, Empire, Omega)
- Test suite execution
- Demo and example runners
- System requirements checking

Can be compiled to standalone executable for Windows (.exe), Linux, and macOS.
"""

import sys
import os
import platform
import subprocess


# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if platform.system() == 'Windows' else 'clear')

def print_banner():
    """Print Pandora banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     ██████╗  █████╗ ███╗   ██╗██████╗  ██████╗ ██████╗  █████╗    ║
    ║     ██╔══██╗██╔══██╗████╗  ██║██╔══██╗██╔═══██╗██╔══██╗██╔══██╗   ║
    ║     ██████╔╝███████║██╔██╗ ██║██║  ██║██║   ██║██████╔╝███████║   ║
    ║     ██╔═══╝ ██╔══██║██║╚██╗██║██║  ██║██║   ██║██╔══██╗██╔══██║   ║
    ║     ██║     ██║  ██║██║ ╚████║██████╔╝╚██████╔╝██║  ██║██║  ██║   ║
    ║     ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ║
    ║                                                               ║
    ║          Quantum Interface with Hamiltonian Support          ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"    Platform: {platform.system()} {platform.release()}")
    print(f"    Python: {sys.version.split()[0]}")
    print()

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    dependencies = {
        'numpy': 'NumPy (matrix operations)',
        'scipy': 'SciPy (matrix exponential)'
    }
    
    missing = []
    for module, description in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {description}")
        except ImportError:
            print(f"  ✗ {description} - MISSING")
            missing.append(module)
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install numpy scipy")
        return False
    
    print("✓ All dependencies satisfied\n")
    return True

def check_hardware():
    """Check hardware capabilities."""
    print("Hardware Detection:")
    
    # Check for NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
        has_nvidia = result.returncode == 0
    except:
        has_nvidia = False
    print(f"  NVIDIA GPU: {'✓ Detected' if has_nvidia else '✗ Not detected (optional)'}")
    
    # Check for Intel CPU
    try:
        if platform.system() == "Windows":
            output = subprocess.check_output("wmic cpu get name", shell=True, timeout=2)
            has_intel = b'Intel' in output
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                has_intel = "Intel" in f.read()
        else:
            has_intel = False
    except:
        has_intel = False
    print(f"  Intel CPU: {'✓ Detected' if has_intel else '✗ Not detected (optional)'}")
    
    print()

def show_menu():
    """Display main menu."""
    print("=" * 65)
    print("                         MAIN MENU")
    print("=" * 65)
    print()
    print("  1. Run Quantum Profile Tests")
    print("  2. Launch Quantum Hamiltonian Demo")
    print("  3. Run Security Integration Example")
    print("  4. Interactive Quantum Processor Session")
    print("  5. Select and Launch Quantum Profile")
    print("  6. System Information")
    print("  0. Exit")
    print()
    print("=" * 65)

def run_tests():
    """Run the comprehensive test suite."""
    print("\n" + "=" * 65)
    print("Running Quantum Profile Test Suite...")
    print("=" * 65 + "\n")
    
    try:
        import test_quantum_hamiltonian
        test_quantum_hamiltonian.run_all_tests()
        input("\nPress Enter to continue...")
    except Exception as e:
        print(f"Error running tests: {e}")
        input("\nPress Enter to continue...")

def run_demo():
    """Run the demonstration script."""
    print("\n" + "=" * 65)
    print("Launching Quantum Hamiltonian Demonstration...")
    print("=" * 65 + "\n")
    
    try:
        import demo_quantum_hamiltonian
        demo_quantum_hamiltonian.main()
        input("\nPress Enter to continue...")
    except Exception as e:
        print(f"Error running demo: {e}")
        input("\nPress Enter to continue...")

def run_security_example():
    """Run the security integration example."""
    print("\n" + "=" * 65)
    print("Launching Quantum Security Integration Example...")
    print("=" * 65 + "\n")
    
    try:
        import example_quantum_security_integration
        example_quantum_security_integration.main()
        input("\nPress Enter to continue...")
    except Exception as e:
        print(f"Error running security example: {e}")
        input("\nPress Enter to continue...")

def interactive_session():
    """Start an interactive quantum processor session."""
    print("\n" + "=" * 65)
    print("Interactive Quantum Processor Session")
    print("=" * 65 + "\n")
    
    try:
        from quantum_virtual_processor import QuantumVirtualProcessor
        import numpy as np
        
        print("Creating QuantumVirtualProcessor with 4 qubits...")
        qvp = QuantumVirtualProcessor(qubits=4, profile='alternative')
        
        print(f"Active profile: {qvp.profile_manager.get_active_profile_name()}")
        print(f"Initial energy: {qvp.compute_energy():.4f}")
        
        print("\nAvailable commands:")
        print("  - Add term: qvp.add_hamiltonian_term(coefficient, pauli_string)")
        print("  - Compute energy: qvp.compute_energy()")
        print("  - Time evolve: qvp.time_evolve(time)")
        print("  - Switch profile: qvp.switch_profile('castle')")
        print("  - Get summary: qvp.get_summary()")
        
        print("\nEntering interactive Python console...")
        print("Type 'exit()' to return to main menu\n")
        
        import code
        code.interact(local={'qvp': qvp, 'np': np})
        
    except Exception as e:
        print(f"Error in interactive session: {e}")
        input("\nPress Enter to continue...")

def select_profile():
    """Select and launch a specific quantum profile."""
    print("\n" + "=" * 65)
    print("Quantum Profile Selection")
    print("=" * 65 + "\n")
    
    profiles = {
        '1': ('alternative', 'Superposition exploration'),
        '2': ('castle', 'Defensive stability'),
        '3': ('hive', 'Collective behavior'),
        '4': ('empire', 'Hierarchical control'),
        '5': ('omega', 'Balanced configuration')
    }
    
    print("Available Profiles:")
    for key, (name, desc) in profiles.items():
        print(f"  {key}. {name.capitalize():12} - {desc}")
    print()
    
    choice = input("Select profile (1-5): ").strip()
    
    if choice in profiles:
        profile_name, description = profiles[choice]
        print(f"\nLaunching {profile_name.capitalize()} profile...")
        
        try:
            from quantum_virtual_processor import QuantumVirtualProcessor
            
            qvp = QuantumVirtualProcessor(qubits=4, profile=profile_name)
            print(f"\n✓ {profile_name.capitalize()} profile loaded")
            print(f"  Description: {description}")
            print(f"  Qubits: 4")
            print(f"  Initial energy: {qvp.compute_energy():.4f}")
            
            h = qvp.get_hamiltonian()
            print(f"  Hamiltonian terms: {h.get_num_terms()}")
            
            ground_energy, _ = h.get_ground_state()
            print(f"  Ground state energy: {ground_energy:.4f}")
            
            input("\nPress Enter to continue...")
        except Exception as e:
            print(f"Error launching profile: {e}")
            input("\nPress Enter to continue...")
    else:
        print("Invalid selection")
        input("\nPress Enter to continue...")

def show_system_info():
    """Display system information."""
    print("\n" + "=" * 65)
    print("System Information")
    print("=" * 65 + "\n")
    
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    
    print("\nInstalled Modules:")
    modules = ['numpy', 'scipy']
    for module in modules:
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  {module}: {version}")
        except ImportError:
            print(f"  {module}: Not installed")
    
    print("\nQuantum Profiles:")
    try:
        from quantum_profiles import QuantumProfileManager
        manager = QuantumProfileManager(default_profile='alternative', n_qubits=3)
        profiles = manager.list_profiles()
        for profile in profiles:
            print(f"  ✓ {profile}")
    except Exception as e:
        print(f"  Error loading profiles: {e}")
    
    input("\nPress Enter to continue...")

def main():
    """Main launcher function."""
    # Check if running as compiled executable
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        application_path = os.path.dirname(sys.executable)
    else:
        # Running as script
        application_path = os.path.dirname(os.path.abspath(__file__))
    
    os.chdir(application_path)
    
    while True:
        clear_screen()
        print_banner()
        
        # First-time checks
        if not hasattr(main, 'checked'):
            check_hardware()
            if not check_dependencies():
                print("\nPlease install missing dependencies before continuing.")
                input("Press Enter to exit...")
                sys.exit(1)
            main.checked = True
        
        show_menu()
        
        choice = input("Enter your choice: ").strip()
        
        if choice == '1':
            run_tests()
        elif choice == '2':
            run_demo()
        elif choice == '3':
            run_security_example()
        elif choice == '4':
            interactive_session()
        elif choice == '5':
            select_profile()
        elif choice == '6':
            show_system_info()
        elif choice == '0':
            print("\nExiting Pandora Quantum System...")
            print("Thank you for using Pandora!\n")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please try again.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)
