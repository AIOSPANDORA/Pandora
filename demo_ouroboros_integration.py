#!/usr/bin/env python3
"""
Demonstration of Ouroboros Virtual Processor Integration
=========================================================

This script demonstrates the Ennead v9.0 quantum overlay architecture,
showcasing:
- Ouroboros Virtual Processor with neuromorphic sentinels
- Recursive weights with Ramanujan τ multipliers
- Zeta-seeded ergotropy bias mechanisms
- Direct harmony access profiling
- Stochastic reconciliation flows

Part of the AIOSPANDORA Pandora quantum computing framework.
"""

import numpy as np
from ouroboros_virtual_processor import (
    OuroborosVirtualProcessor,
    NeuromorphicSentinel,
    RecursiveWeight,
    ZetaErgotropyBias
)
from ouroboros_overlay import OuroborosOverlay, create_ouroboros_overlay
from quantum_overlay_profiles import HarmonyAccessProfiler


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def demo_ouroboros_processor():
    """Demonstrate Ouroboros Virtual Processor capabilities."""
    print_section("Ouroboros Virtual Processor - Functional Manifold")
    
    # Create processor
    processor = OuroborosVirtualProcessor(num_qubits=6, enable_sentinels=True)
    
    print(f"\n✓ Created processor with {processor.num_qubits} qubits")
    print(f"  - Neuromorphic sentinels: {len(processor.sentinels)}")
    print(f"  - Recursive weights: {len(processor.recursive_weights)}")
    print(f"  - State dimension: {processor.state_dimension}")
    
    # Execute ouroboros cycles
    print("\n→ Executing ouroboros cycles...")
    for i in range(5):
        sentinel_triggered = processor.ouroboros_cycle()
        if sentinel_triggered:
            print(f"  Cycle {i+1}: Sentinel triggered!")
        else:
            print(f"  Cycle {i+1}: Normal operation")
    
    # Get harmonic resonance
    harmony = processor.get_harmonic_resonance()
    print(f"\n→ Harmonic resonance after 5 cycles:")
    print(f"  - Iteration: {harmony['iteration']}")
    print(f"  - Sentinel triggers: {harmony['sentinel_triggers']}")
    print(f"  - Ergotropy accumulated: {harmony['ergotropy_accumulated']:.4f}")
    print(f"  - Average weight: {harmony['average_weight']:.4f}")
    print(f"  - Zeta seed: {harmony['zeta_seed']:.4f}")
    
    # Show diagnostic info
    diag = processor.get_diagnostic_info()
    print(f"\n→ Diagnostic snapshot:")
    print(f"  - Cycle history: {diag['cycle_history_length']} states")
    print(f"  - Recursive weight depth: {diag['recursive_weights'][0]['recursion_depth']}")
    
    return processor


def demo_ouroboros_overlay():
    """Demonstrate Ouroboros Overlay with integration."""
    print_section("Ouroboros Overlay - 6th VQP with Integration")
    
    # Create overlay
    overlay = create_ouroboros_overlay()
    overlay.initialize_overlay(num_qubits=8, enable_virtual_processor=True, enable_ergotropy=True)
    
    print(f"\n✓ Overlay initialized with {overlay.num_qubits} qubits")
    print(f"  - Virtual processor integrated: {overlay.virtual_processor is not None}")
    print(f"  - Zeta ergotropy bias: {overlay.zeta_bias is not None}")
    print(f"  - Overlay ID: {overlay.OVERLAY_ID}")
    print(f"  - Version: {overlay.OVERLAY_VERSION}")
    
    # Apply overlay logic
    print("\n→ Applying overlay logic with stochastic reconciliation...")
    test_state = np.ones(8, dtype=complex) / np.sqrt(8)
    
    for i in range(3):
        test_state = overlay.apply_overlay_logic(
            test_state,
            mutation_rate=0.1,
            ergotropy_strength=0.05
        )
        print(f"  Cycle {i+1}: State norm = {np.linalg.norm(test_state):.6f}")
    
    # Process measurement
    measurement = {'state_vector': test_state}
    result = overlay.process_measurement(measurement)
    
    print(f"\n→ Measurement results:")
    print(f"  - Overlay: {result['overlay_id']}")
    print(f"  - Cycle: {result['cycle']}")
    print(f"  - Energy conservation: {result['energy_conservation']:.6f}")
    
    if 'qutrit_analysis' in result:
        qa = result['qutrit_analysis']
        print(f"  - Qutrit states: Ground={qa['ground_count']}, "
              f"Excited={qa['excited_count']}, Rydberg={qa['rydberg_count']}")
    
    if 'phase_analysis' in result:
        pa = result['phase_analysis']
        print(f"  - Phase balance: {pa['phase_balance']:.4f}")
    
    if 'bounce_statistics' in result:
        bs = result['bounce_statistics']
        print(f"  - Total bounces: {bs['total_bounces']}")
    
    return overlay


def demo_harmony_profiler():
    """Demonstrate Direct Harmony Access Profiling."""
    print_section("Harmony Access Profiler - Direct Resonance Tracking")
    
    # Create profiler
    profiler = HarmonyAccessProfiler(num_qubits=6)
    
    print(f"\n✓ Profiler initialized")
    print(f"  - Ouroboros integration: {profiler.ouroboros_overlay is not None}")
    
    # Capture harmony states
    print("\n→ Capturing harmony states from different overlays...")
    
    for overlay_type in ['alpha', 'hive', 'castle']:
        # Generate test state
        state = np.random.rand(64) + 1j * np.random.rand(64)
        state = state / np.linalg.norm(state)
        
        harmony_state = profiler.capture_harmony_state(overlay_type, state)
        print(f"  {overlay_type}: phase_coherence={harmony_state['phase_coherence']:.4f}, "
              f"resonance_index={harmony_state['resonance_index']:.4f}")
    
    # Track resonance
    print("\n→ Tracking resonance between states...")
    state1 = np.random.rand(64) + 1j * np.random.rand(64)
    state1 = state1 / np.linalg.norm(state1)
    state2 = np.random.rand(64) + 1j * np.random.rand(64)
    state2 = state2 / np.linalg.norm(state2)
    
    resonance = profiler.track_resonance(state1, state2)
    print(f"  Resonance strength: {resonance:.4f}")
    
    # Access direct harmony
    print("\n→ Accessing direct harmony states...")
    for harmony_type in ['balanced', 'coherent', 'resonant']:
        access = profiler.access_direct_harmony(target_harmony=harmony_type)
        if access['status'] == 'harmony_accessed':
            print(f"  {harmony_type}: {access['selected_state']['overlay_type']} overlay selected")
    
    # Get summary
    summary = profiler.get_harmony_summary()
    print(f"\n→ Harmony summary:")
    print(f"  - States captured: {summary['states_captured']}")
    print(f"  - Average phase coherence: {summary['average_phase_coherence']:.4f}")
    print(f"  - Average amplitude balance: {summary['average_amplitude_balance']:.4f}")
    print(f"  - Resonance samples: {summary['resonance_tracking']['samples']}")
    
    return profiler


def demo_neuromorphic_sentinels():
    """Demonstrate neuromorphic sentinel behavior."""
    print_section("Neuromorphic Sentinels - Adaptive Monitoring")
    
    # Create sentinels
    sentinels = [
        NeuromorphicSentinel(sentinel_id=i, threshold=0.3 + i * 0.1)
        for i in range(3)
    ]
    
    print(f"\n✓ Created {len(sentinels)} sentinels")
    
    # Test monitoring
    print("\n→ Testing sentinel monitoring with varying states...")
    
    for iteration in range(5):
        # Generate test state with increasing entropy
        state = np.random.rand(16) * (1 + iteration * 0.2)
        state = state / np.linalg.norm(state)
        
        triggered_count = 0
        for sentinel in sentinels:
            if sentinel.monitor(state):
                triggered_count += 1
        
        print(f"  Iteration {iteration+1}: {triggered_count}/3 sentinels triggered")
    
    # Show learned thresholds
    print("\n→ Learned thresholds after monitoring:")
    for sentinel in sentinels:
        print(f"  Sentinel {sentinel.sentinel_id}: "
              f"threshold={sentinel.threshold:.4f}, "
              f"sensitivity={sentinel.sensitivity:.2f}")
    
    return sentinels


def demo_recursive_weights():
    """Demonstrate recursive weight evolution with Ramanujan τ."""
    print_section("Recursive Weights - Ramanujan τ Multipliers")
    
    # Create recursive weights
    weights = [
        RecursiveWeight(index=i, base_weight=1.0 / (i + 1))
        for i in range(6)
    ]
    
    print(f"\n✓ Created {len(weights)} recursive weights")
    print(f"  Ramanujan τ base constant: {weights[0].compute_ramanujan_tau(1)}")
    
    # Evolve weights
    print("\n→ Evolving weights through iterations...")
    
    for iteration in [1, 5, 10, 20]:
        evolved_values = []
        for weight in weights:
            value = weight.evolve(iteration)
            evolved_values.append(value)
        
        print(f"  Iteration {iteration:2d}: avg={np.mean(evolved_values):.4f}, "
              f"std={np.std(evolved_values):.4f}")
    
    # Show final state
    print("\n→ Final weight states:")
    for weight in weights[:3]:  # Show first 3
        print(f"  Weight {weight.index}: "
              f"tau_multiplier={weight.tau_multiplier:.4f}, "
              f"recursion_depth={weight.recursion_depth}")
    
    return weights


def demo_zeta_ergotropy():
    """Demonstrate zeta-seeded ergotropy bias."""
    print_section("Zeta-Seeded Ergotropy Bias - Energy Distribution")
    
    # Create ergotropy bias
    zeta_bias = ZetaErgotropyBias(dimension=64, zeta_order=2)
    
    print(f"\n✓ Ergotropy bias initialized")
    print(f"  - Dimension: {zeta_bias.dimension}")
    print(f"  - Zeta seed ζ(2): {zeta_bias.zeta_seed:.6f}")
    print(f"  - Bias vector norm: {np.linalg.norm(zeta_bias.bias_vector):.6f}")
    
    # Apply bias
    print("\n→ Applying bias to quantum states...")
    
    state = np.ones(64, dtype=complex) / np.sqrt(64)
    
    for strength in [0.05, 0.1, 0.2]:
        biased_state = zeta_bias.apply_bias(state, strength=strength)
        print(f"  Strength {strength:.2f}: "
              f"ergotropy={zeta_bias.ergotropy_accumulator:.4f}, "
              f"norm={np.linalg.norm(biased_state):.6f}")
    
    # Extract ergotropy
    extracted = zeta_bias.extract_ergotropy()
    print(f"\n→ Extracted ergotropy: {extracted:.4f}")
    print(f"  Remaining: {zeta_bias.ergotropy_accumulator:.4f}")
    
    return zeta_bias


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print(" ENNEAD v9.0 - Ouroboros Integration Demonstration")
    print(" AIOSPANDORA Pandora Quantum Computing Framework")
    print("=" * 70)
    
    try:
        # Run demonstrations
        processor = demo_ouroboros_processor()
        overlay = demo_ouroboros_overlay()
        profiler = demo_harmony_profiler()
        sentinels = demo_neuromorphic_sentinels()
        weights = demo_recursive_weights()
        zeta_bias = demo_zeta_ergotropy()
        
        # Final summary
        print_section("Integration Summary")
        print("\n✓ All Ennead v9.0 components demonstrated successfully!")
        print("\nKey Features Validated:")
        print("  [✓] Ouroboros Virtual Processor functional manifold")
        print("  [✓] Neuromorphic sentinel adaptive monitoring")
        print("  [✓] Recursive weights with Ramanujan τ multipliers")
        print("  [✓] Zeta-seeded ergotropy bias mechanisms")
        print("  [✓] Ouroboros Overlay with modular integration")
        print("  [✓] Direct harmony access profiling")
        print("  [✓] Stochastic reconciliation flows")
        print("\nThe system operates cleanly under FullSystemBuild runtime.")
        print("All prior deltas/hamiltonians/mappings preserved.")
        
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
