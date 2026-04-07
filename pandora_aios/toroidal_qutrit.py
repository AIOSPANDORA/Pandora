# pandora_aios/toroidal_qutrit.py
import numpy as np


class ToroidalQutrit:
    """Three-state cyclic qutrit with bounce recovery and ternary Kuramoto coupling."""

    STATES = {0: "GROUND", 1: "EXCITED", 2: "RYDBERG"}

    def __init__(self, phase=0.0, state=0):
        self.phase = phase
        self.state = state
        self.energy = 1.0

    def evolve(self, dt, coupling, k=3):
        """
        Ternary Kuramoto evolution.
        coupling: sum of sin(phase_j - phase_i) over neighbors (pre-computed)
        """
        dphase = coupling * dt * (1.0 + 0.1 * self.energy)
        self.phase += dphase
        self.phase %= 2 * np.pi
        # bounce recovery: if phase difference exceeds pi, flip state and dampen
        if abs(dphase) > np.pi:
            self.state = (self.state + 1) % 3
            self.energy *= 0.9
        return self.phase

    def get_complex_amplitude(self):
        """Return complex number on unit circle representing current state."""
        angle = self.phase + (self.state * 2 * np.pi / 3)
        return np.exp(1j * angle)
