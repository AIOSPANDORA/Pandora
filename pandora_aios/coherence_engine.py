# pandora_aios/coherence_engine.py
from pandora_aios.quantum_virtual_processor import QuantumVirtualProcessor
from pandora_aios.hebbian import HebbianWeightUpdate
import numpy as np


class CoherenceEngine:
    """Toroidal Qutrit + E8E8 Coherence Engine orchestrator."""

    def __init__(self, dimensions=248, qutrits=64):
        self.dim = dimensions
        self.qutrits = qutrits
        self.qvp = QuantumVirtualProcessor(num_qutrits=qutrits)
        self.hebbian = HebbianWeightUpdate(dim=2 * dimensions)  # 496

    def initialize(self):
        """Activate qutrit layer and pre-compute E8E8 projection matrix."""
        self.qvp.activate_qutrit_layer(self.qutrits)
        self.qvp.e8e8_project()  # pre-compute projection matrix

    def update(self, dt=0.05, k=5):
        """Run one evolution step and apply Hebbian update."""
        proj = self.qvp.step(dt, k)  # returns 496-dim vector
        phases = np.array([q.phase for q in self.qvp.qutrits])
        self.hebbian.update(proj, phases=phases)
        return proj

    def get_coherence(self):
        """Return the current Kuramoto order parameter."""
        return self.qvp.get_coherence()
