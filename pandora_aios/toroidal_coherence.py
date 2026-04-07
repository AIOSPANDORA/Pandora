# pandora_aios/toroidal_coherence.py
import numpy as np


class ToroidalCoherence:
    """Toroidal coherence tracker using Kuramoto order parameter over cyclic neighbors."""

    def __init__(self, phases, k=5):
        """
        Args:
            phases: array of oscillator phases
            k: number of nearest neighbors on each side for coupling
        """
        self.phases = np.asarray(phases, dtype=float)
        self.k = k

    def kuramoto_order_parameter(self):
        """
        Compute the Kuramoto order parameter r = |1/N * sum(exp(i * phase_j))|.
        Returns a float in [0, 1]. r ~ 1 means full synchronization.
        """
        n = len(self.phases)
        if n == 0:
            return 0.0
        return float(np.abs(np.mean(np.exp(1j * self.phases))))

    def coupling_field(self):
        """
        Compute the coupling field for each oscillator:
        coupling_i = (1 / 2k) * sum_{j in neighbors} sin(phase_j - phase_i)
        """
        n = len(self.phases)
        coupling = np.zeros(n)
        for i in range(1, self.k + 1):
            coupling += np.sin(np.roll(self.phases, -i) - self.phases)
            coupling += np.sin(np.roll(self.phases, i) - self.phases)
        if self.k > 0:
            coupling /= (2 * self.k)
        return coupling
