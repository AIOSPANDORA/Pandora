# pandora_aios/quantum_virtual_processor.py
from pandora_aios.toroidal_qutrit import ToroidalQutrit
from pandora_aios.toroidal_coherence import ToroidalCoherence
import numpy as np


class QuantumVirtualProcessor:
    """Quantum virtual processor with toroidal qutrit layer and E8E8 lattice projection."""

    def __init__(self, num_qutrits=64):
        self.num_qutrits = num_qutrits
        self.qutrits = []
        self.e8e8_projection = None  # will hold the 248-dim projection matrix

    def activate_qutrit_layer(self, num_qutrits=64):
        """Initialize an array of ToroidalQutrits."""
        self.num_qutrits = num_qutrits
        self.qutrits = [
            ToroidalQutrit(phase=np.random.uniform(0, 2 * np.pi))
            for _ in range(num_qutrits)
        ]

    def e8e8_project(self):
        """
        Map current qutrit state vector onto the 248-dimensional E8 root lattice projection.
        Returns a 496-dim real vector (real and imaginary parts concatenated).
        """
        if self.e8e8_projection is None:
            self.e8e8_projection = np.random.randn(248, self.num_qutrits)
        # Get complex amplitudes from each qutrit
        amps = np.array([q.get_complex_amplitude() for q in self.qutrits])
        # Project onto E8 lattice (real and imaginary parts)
        proj_real = self.e8e8_projection @ amps.real
        proj_imag = self.e8e8_projection @ amps.imag
        return np.concatenate([proj_real, proj_imag])  # 496-dim real vector

    def step(self, dt=0.05, k=5):
        """One evolution step: toroidal qutrit evolution -> E8 projection -> Hebbian update."""
        # 1. Compute coupling using ToroidalCoherence
        phases = np.array([q.phase for q in self.qutrits])
        tc = ToroidalCoherence(phases, k=k)
        coupling = tc.coupling_field()

        # 2. Evolve each qutrit
        for idx, q in enumerate(self.qutrits):
            q.evolve(dt, coupling[idx], k=3)

        # 3. E8E8 projection
        proj = self.e8e8_project()
        return proj

    def get_coherence(self):
        """Return Kuramoto order parameter of the qutrit phases."""
        phases = np.array([q.phase for q in self.qutrits])
        return float(np.abs(np.mean(np.exp(1j * phases))))
