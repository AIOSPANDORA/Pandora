# pandora_aios/hebbian.py
import numpy as np
from pandora_aios.toroidal_coherence import ToroidalCoherence


class HebbianWeightUpdate:
    """Hebbian weight update fused with E8E8 projection and Kuramoto coherence normalization."""

    def __init__(self, dim=496, eta=0.01, decay=0.001, default_num_phases=64):
        self.W = np.eye(dim) * 0.1
        self.eta = eta
        self.decay = decay
        self.default_num_phases = default_num_phases

    def update(self, proj_vector, phases=None):
        """
        Update weights using the E8E8-projected correlation matrix.

        Args:
            proj_vector: 496-dim real vector from E8E8 projection.
            phases: optional array of qutrit phases for Kuramoto normalization.
                    If None, a default uniform distribution is used.
        """
        # Compute correlation matrix
        corr = np.outer(proj_vector, proj_vector)

        # Normalize using Kuramoto order parameter
        if phases is None:
            phases = np.random.rand(self.default_num_phases) * 2 * np.pi
        order_param = ToroidalCoherence(phases, k=5).kuramoto_order_parameter()
        if order_param > 0:
            corr /= order_param

        # Hebbian update
        self.W += self.eta * corr - self.decay * self.W

        # Prune using qutrit bounce energy (norm of proj_vector as proxy)
        prune_mask = np.abs(self.W) < 0.01 * np.linalg.norm(proj_vector)
        self.W[prune_mask] = 0.0
        return self.W
