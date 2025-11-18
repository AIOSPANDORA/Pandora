"""
Empire Quantum Virtual Processor - Usage Example
=================================================

This script demonstrates how to use the Empire quantum processor profile
with Pandora's API.
"""

from quantum_profiles import get_profile, list_profiles


def main():
    print("=" * 70)
    print("Empire Quantum Virtual Processor - Usage Demonstration")
    print("=" * 70)
    print()
    
    # List available profiles
    print("1. Available Quantum Profiles:")
    profiles = list_profiles()
    for profile in profiles:
        print(f"   - {profile}")
    print()
    
    # Create an Empire processor with default settings
    print("2. Creating Empire processor with default settings...")
    empire = get_profile('empire')
    print(f"   {empire}")
    print(f"   Stats: {empire.get_empire_stats()}")
    print()
    
    # Create a custom Empire processor
    print("3. Creating custom Empire processor...")
    empire_custom = get_profile(
        'empire',
        control_qubits=6,
        grid_size=(3, 3),
        hive_qubits=12
    )
    print(f"   {empire_custom}")
    print(f"   Stats: {empire_custom.get_empire_stats()}")
    print()
    
    # Demonstrate gate application to entire empire
    print("4. Applying Hadamard gate to entire Empire...")
    empire.apply_gate_to_empire("H", 0)
    print("   ✓ Gate applied to control lattice and all hives")
    print()
    
    # Demonstrate gate application to control lattice only
    print("5. Applying X gate to control lattice only...")
    empire.apply_gate_to_control_lattice("X", 1)
    print("   ✓ Gate applied to control lattice only")
    print()
    
    # Demonstrate gate application to specific hive
    print("6. Applying Y gate to specific hive (0, 1)...")
    empire.apply_gate_to_hive((0, 1), "Y", 2)
    print("   ✓ Gate applied to hive (0, 1)")
    print()
    
    # List all hives
    print("7. Available hives in the grid:")
    for hive_id in empire.list_hives():
        hive = empire.get_hive(hive_id)
        print(f"   - {hive}")
    print()
    
    # Demonstrate measurements
    print("8. Measuring the entire Empire...")
    results = empire.measure_empire()
    print(f"   Control lattice measurement: {results['control']}")
    print(f"   Number of hive measurements: {len(results['hives'])}")
    print()
    
    print("9. Measuring control lattice only...")
    control_result = empire.measure_control_lattice()
    print(f"   Result: {control_result}")
    print()
    
    print("10. Measuring specific hive (1, 0)...")
    hive_result = empire.measure_hive((1, 0))
    print(f"    Result: {hive_result}")
    print()
    
    # Demonstrate dynamic expansion
    print("11. Dynamically expanding the hive grid...")
    print(f"    Current grid size: {empire.grid_size}")
    print(f"    Current total hives: {len(empire.hive_grid)}")
    empire.expand_grid((3, 4))
    print(f"    New grid size: {empire.grid_size}")
    print(f"    New total hives: {len(empire.hive_grid)}")
    print()
    
    print("12. Expanding the control lattice...")
    print(f"    Current control qubits: {empire.control_qubits}")
    empire.expand_control_lattice(8)
    print(f"    New control qubits: {empire.control_qubits}")
    print()
    
    # Demonstrate addon support
    print("13. Registering and executing an addon...")
    
    class ExampleAddon:
        def __init__(self):
            self.name = "quantum_optimizer"
        
        def execute(self, empire, *args, **kwargs):
            stats = empire.get_empire_stats()
            return f"Optimized Empire with {stats['total_qubits']} total qubits"
    
    addon = ExampleAddon()
    empire.register_addon(addon)
    result = empire.execute_addon("quantum_optimizer")
    print(f"    Addon result: {result}")
    print()
    
    # Final statistics
    print("14. Final Empire statistics:")
    final_stats = empire.get_empire_stats()
    for key, value in final_stats.items():
        print(f"    {key}: {value}")
    print()
    
    print("=" * 70)
    print("Demonstration Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
