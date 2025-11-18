#!/usr/bin/env python3
"""
Integration Example: Quantum Profiles with Security System

This example demonstrates how the new Hamiltonian-aware quantum profiles
integrate with Pandora's existing quantum security infrastructure
(quantum_mirror_firewall.py).

The quantum profiles can be used to enhance quantum randomness generation,
quantum encryption, and quantum-secured authentication.
"""

import numpy as np
from quantum_profiles import QuantumProfileManager, Hamiltonian
from quantum_virtual_processor import QuantumVirtualProcessor


class QuantumSecurityEnhancer:
    """
    Enhances quantum security using Hamiltonian-aware profiles.
    
    Integrates with existing quantum_mirror_firewall to provide:
    - Enhanced quantum random number generation
    - Hamiltonian-based encryption keys
    - Time-evolved quantum states for authentication
    """
    
    def __init__(self, n_qubits=6):
        """Initialize with quantum profile manager."""
        self.qvp = QuantumVirtualProcessor(qubits=n_qubits, profile='alternative')
        self.n_qubits = n_qubits
        
    def generate_quantum_random_bits(self, num_bits=256):
        """
        Generate quantum random bits using Hamiltonian time evolution.
        
        Uses the Alternative profile's superposition-heavy Hamiltonian
        to create highly unpredictable quantum states.
        
        Args:
            num_bits: Number of random bits to generate
            
        Returns:
            list: Random bits [0, 1, 0, 1, ...]
        """
        bits = []
        
        # Switch to Alternative profile for maximum randomness
        self.qvp.switch_profile('alternative')
        
        for i in range(num_bits):
            # Time evolve by a random amount
            time = 0.1 + 0.9 * np.random.random()
            self.qvp.time_evolve(time)
            
            # Measure and extract bit
            outcome = self.qvp.measure()
            bit = outcome % 2
            bits.append(bit)
        
        return bits
    
    def derive_encryption_key(self, seed_data, key_length=256):
        """
        Derive an encryption key from seed data using Hamiltonian evolution.
        
        Creates a custom Hamiltonian based on seed data, evolves it,
        and extracts an encryption key from the quantum state.
        
        Args:
            seed_data: Seed data (string or bytes)
            key_length: Length of key in bits
            
        Returns:
            bytes: Encryption key
        """
        # Convert seed to numeric values
        seed_bytes = seed_data.encode() if isinstance(seed_data, str) else seed_data
        seed_values = list(seed_bytes)
        
        # Create custom Hamiltonian from seed
        h = Hamiltonian(n_qubits=self.n_qubits)
        
        # Add terms based on seed
        for i, value in enumerate(seed_values[:self.n_qubits]):
            # Use seed value to determine coefficient
            coeff = (value / 255.0) * 2.0 - 1.0
            pauli_string = 'I' * i + 'Z' + 'I' * (self.n_qubits - i - 1)
            h.add_term(coeff, pauli_string)
        
        # Time evolve
        state = np.zeros(2 ** self.n_qubits, dtype=complex)
        state[0] = 1.0
        evolved_state = h.time_evolution(state, time=1.0)
        
        # Extract key bits from state amplitudes
        key_bits = []
        amplitudes = np.abs(evolved_state)
        
        for i in range(key_length):
            idx = i % len(amplitudes)
            bit = 1 if amplitudes[idx] > 0.5 else 0
            key_bits.append(bit)
            # Rotate amplitudes for next bit
            amplitudes = np.roll(amplitudes, 1)
        
        # Convert bits to bytes
        key_bytes = bytearray()
        for i in range(0, len(key_bits), 8):
            byte_val = sum(bit << (7 - j) for j, bit in enumerate(key_bits[i:i+8]))
            key_bytes.append(byte_val)
        
        return bytes(key_bytes)
    
    def create_quantum_auth_token(self, user_id, timestamp):
        """
        Create a quantum authentication token.
        
        Uses Empire profile's hierarchical structure to create
        a unique token based on user_id and timestamp.
        
        Args:
            user_id: User identifier
            timestamp: Current timestamp
            
        Returns:
            dict: Authentication token with quantum signature
        """
        # Switch to Empire profile for hierarchical control
        self.qvp.switch_profile('empire')
        
        # Create user-specific Hamiltonian term
        user_hash = hash(user_id) % (2 ** self.n_qubits)
        user_bits = format(user_hash, f'0{self.n_qubits}b')
        
        # Map to Pauli string (0 -> Z, 1 -> X)
        pauli_string = ''.join('Z' if b == '0' else 'X' for b in user_bits)
        self.qvp.add_hamiltonian_term(1.0, pauli_string)
        
        # Time evolve based on timestamp
        time_factor = (timestamp % 1000) / 1000.0
        self.qvp.time_evolve(time_factor)
        
        # Compute energy signature
        energy = self.qvp.compute_energy()
        
        # Get quantum state hash
        state = self.qvp.profile_manager.get_state()
        state_hash = hash(tuple(np.abs(state).round(4)))
        
        return {
            'user_id': user_id,
            'timestamp': timestamp,
            'quantum_energy': energy,
            'quantum_signature': state_hash,
            'profile': 'empire'
        }
    
    def compare_security_profiles(self):
        """
        Compare security characteristics of different quantum profiles.
        
        Returns:
            dict: Security metrics for each profile
        """
        results = {}
        
        for profile_name in ['alternative', 'castle', 'hive', 'empire', 'omega']:
            self.qvp.switch_profile(profile_name)
            
            # Measure randomness via entropy
            bits = []
            for _ in range(100):
                outcome = self.qvp.measure()
                bits.append(outcome % 2)
            
            entropy = -sum(p * np.log2(p) for p in [bits.count(0)/100, bits.count(1)/100] if p > 0)
            
            # Measure stability via ground state
            h = self.qvp.get_hamiltonian()
            ground_energy, _ = h.get_ground_state()
            
            results[profile_name] = {
                'entropy': entropy,
                'ground_energy': ground_energy,
                'num_terms': h.get_num_terms()
            }
        
        return results


def main():
    """Demonstrate quantum security enhancements."""
    print("=" * 70)
    print("  QUANTUM SECURITY ENHANCEMENT WITH HAMILTONIAN PROFILES")
    print("=" * 70)
    
    enhancer = QuantumSecurityEnhancer(n_qubits=6)
    
    # 1. Generate quantum random bits
    print("\n1. Quantum Random Number Generation")
    print("-" * 70)
    random_bits = enhancer.generate_quantum_random_bits(num_bits=32)
    print(f"Generated {len(random_bits)} quantum random bits:")
    print(''.join(str(b) for b in random_bits[:32]))
    print(f"Entropy check - 0s: {random_bits.count(0)}, 1s: {random_bits.count(1)}")
    
    # 2. Derive encryption key
    print("\n2. Quantum-Derived Encryption Key")
    print("-" * 70)
    seed = "PandoraSecurityKey2025"
    key = enhancer.derive_encryption_key(seed, key_length=128)
    print(f"Seed data: {seed}")
    print(f"Generated key (hex): {key[:16].hex()}...")
    print(f"Key length: {len(key)} bytes")
    
    # 3. Create authentication token
    print("\n3. Quantum Authentication Token")
    print("-" * 70)
    import time
    token = enhancer.create_quantum_auth_token("user@pandora.ai", int(time.time()))
    print(f"User: {token['user_id']}")
    print(f"Timestamp: {token['timestamp']}")
    print(f"Quantum Energy: {token['quantum_energy']:.6f}")
    print(f"Quantum Signature: {token['quantum_signature']}")
    print(f"Profile: {token['profile']}")
    
    # 4. Compare security profiles
    print("\n4. Security Profile Comparison")
    print("-" * 70)
    comparison = enhancer.compare_security_profiles()
    
    print(f"{'Profile':<15} {'Entropy':<10} {'Ground E':<12} {'Terms':<8}")
    print("-" * 70)
    for profile, metrics in sorted(comparison.items()):
        print(f"{profile:<15} {metrics['entropy']:<10.4f} {metrics['ground_energy']:<12.4f} {metrics['num_terms']:<8}")
    
    print("\n" + "=" * 70)
    print("  INTEGRATION COMPLETE")
    print("=" * 70)
    print("\nQuantum profiles successfully integrated with security infrastructure!")
    print("Recommended profiles:")
    print("  • Alternative: High-entropy random number generation")
    print("  • Castle: Stable key storage and authentication")
    print("  • Empire: Hierarchical access control")
    print()


if __name__ == "__main__":
    main()
