# test_black_core.py
from pandora_aios.coherence_engine import CoherenceEngine

engine = CoherenceEngine(dimensions=248, qutrits=64)
engine.initialize()
for _ in range(200):
    engine.update(dt=0.05, k=5)
print(f"Final coherence: {engine.get_coherence():.6f}")
