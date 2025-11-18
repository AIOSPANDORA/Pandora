#!/usr/bin/env python3
"""
Pandora AIOS Demo Script
Demonstrates the improved functionality of core modules.
"""

import sys
import time

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")

def demo_subroutine_ai():
    """Demonstrate SubroutineAI improvements"""
    print_section("SubroutineAI - Environment Detection")
    
    from subroutine_ai import SubroutineAI
    ai = SubroutineAI()
    recommendations = ai.get_all_recommendations()
    
    print(f"OS: {recommendations['os']}")
    print(f"Python: {recommendations['python_version']}")
    print(f"CPU Cores: {recommendations['cpu_count']}")
    print(f"RAM: {recommendations['ram_gb']} GB")
    print(f"GPU Available: {recommendations['has_gpu']}")
    print(f"Optimal Workers: {recommendations['optimal_workers']}")
    print(f"Recommended Cache Size: {recommendations['cache_size']} MB")
    
    if recommendations['warnings']:
        print("\n⚠ Recommendations:")
        for warning in recommendations['warnings']:
            print(f"  - {warning}")

def demo_cache():
    """Demonstrate CalmCacheOptimizer improvements"""
    print_section("CalmCacheOptimizer - Smart Caching")
    
    from calm_cache_optimizer import CalmCacheOptimizer
    cache = CalmCacheOptimizer(max_entries=5, max_age_sec=60)
    
    print("Setting cache values...")
    for i in range(5):
        cache.set(f'key{i}', f'value{i}')
        print(f"  Set key{i} = value{i}")
    
    print("\nRetrieving values...")
    for i in range(7):
        value = cache.get(f'key{i}')
        status = "HIT" if value else "MISS"
        print(f"  Get key{i}: {value} [{status}]")
    
    print("\nCache Statistics:")
    stats = cache.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

def demo_quantum():
    """Demonstrate QuantumVirtualProcessor improvements"""
    print_section("QuantumVirtualProcessor - Quantum Computing")
    
    from quantum_virtual_processor import QuantumVirtualProcessor
    qvp = QuantumVirtualProcessor(qubits=6)
    
    print("Initial quantum state:")
    print(f"  {qvp.get_state_vector()}")
    
    print("\nApplying quantum gates...")
    print("  Hadamard gate on qubit 0 (superposition)")
    qvp.apply_gate('H', 0)
    print("  Pauli-X gate on qubit 1 (bit flip)")
    qvp.apply_gate('X', 1)
    
    print("\nState after gates:")
    print(f"  {qvp.get_state_vector()}")
    
    print("\nGenerating quantum random numbers...")
    for i in range(3):
        rnd = qvp.quantum_random_number(bits=8)
        print(f"  Random {i+1}: {rnd}")

def demo_stoic_adviser():
    """Demonstrate StoicAdviser improvements"""
    print_section("StoicAdviser - Wisdom and Guidance")
    
    from stoic_adviser import StoicAdviser
    adviser = StoicAdviser()
    
    print("General Advice:")
    print(f"  {adviser.advise()}")
    
    print("\nContext-Aware Advice:")
    contexts = ['error', 'success', 'warning', 'startup']
    for context in contexts:
        advice = adviser.advise(context)
        print(f"  [{context.upper()}] {advice}")
    
    print("\nDaily Reflection:")
    print(f"  {adviser.daily_reflection()}")

def demo_config():
    """Demonstrate configuration system"""
    print_section("Configuration System")
    
    from pandora_config import PandoraConfig
    
    print(f"System: {PandoraConfig.SYSTEM_NAME} v{PandoraConfig.VERSION}")
    print(f"Cache: {PandoraConfig.CACHE_MAX_ENTRIES} entries, {PandoraConfig.CACHE_MAX_AGE_SEC}s TTL")
    print(f"Monitoring: Every {PandoraConfig.MONITOR_CHECK_INTERVAL}s")
    print(f"Thresholds: CPU {PandoraConfig.CPU_WARNING_THRESHOLD}%, "
          f"RAM {PandoraConfig.MEMORY_WARNING_THRESHOLD}%, "
          f"Disk {PandoraConfig.DISK_WARNING_THRESHOLD}%")
    
    print("\nValidating configuration...")
    valid, issues = PandoraConfig.validate()
    if valid:
        print("  ✓ Configuration is valid")
    else:
        print("  ✗ Configuration has issues:")
        for issue in issues:
            print(f"    - {issue}")

def main():
    """Run all demos"""
    print("\n" + "█" * 60)
    print("█  PANDORA AIOS v2.5 - IMPROVEMENT DEMONSTRATION")
    print("█" * 60)
    
    try:
        demo_subroutine_ai()
        time.sleep(1)
        
        demo_cache()
        time.sleep(1)
        
        demo_quantum()
        time.sleep(1)
        
        demo_stoic_adviser()
        time.sleep(1)
        
        demo_config()
        
        print("\n" + "=" * 60)
        print("  Demo Complete - All systems operational!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
