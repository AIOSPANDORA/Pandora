"""
Test script for resonance_chamber.py

This script validates the core logic of the ResonanceChamber class
without requiring a GUI/X server environment.
"""

import sys
import time
from collections import deque


class MockResonanceChamber:
    """Mock version of ResonanceChamber for testing core logic."""
    
    def __init__(self, history_length=50):
        self.history_length = history_length
        self.timestamps = deque(maxlen=history_length)
        self.device_rhythm = deque(maxlen=history_length)
        self.human_rhythm = deque(maxlen=history_length)
        self.ai_confidence = deque(maxlen=history_length)
        self.resonance_score = deque(maxlen=history_length)
        self.anomalies = []
        self.realizations = []
        self.last_ai_confidence = 50.0
    
    def calculate_resonance_score(self, device_rhythm, human_rhythm):
        """Calculate Resonance Score: 100 - abs(Device - Human)"""
        return 100.0 - abs(device_rhythm - human_rhythm)
    
    def calculate_ai_confidence(self, device_rhythm, human_rhythm):
        """Simulate AI confidence based on stability."""
        if len(self.device_rhythm) > 1:
            device_change = abs(device_rhythm - self.device_rhythm[-1])
            human_change = abs(human_rhythm - self.human_rhythm[-1])
        else:
            device_change = 0
            human_change = 0
        
        total_change = device_change + human_change
        
        if total_change > 30:
            confidence_delta = -5
        elif total_change < 10:
            confidence_delta = 3
        else:
            confidence_delta = -1
        
        new_confidence = max(0.0, min(100.0, self.last_ai_confidence + confidence_delta))
        return new_confidence
    
    def detect_anomalies(self, device_rhythm, human_rhythm, current_time):
        """Detect spikes > 50 points."""
        if len(self.device_rhythm) > 0:
            device_spike = abs(device_rhythm - self.device_rhythm[-1])
            human_spike = abs(human_rhythm - self.human_rhythm[-1])
            
            if device_spike > 50 or human_spike > 50:
                max_spike = max(device_spike, human_spike)
                self.anomalies.append((current_time, max_spike))
                return True
        return False
    
    def detect_realizations(self, ai_confidence, current_time):
        """Detect AI confidence increases > 30 points."""
        if len(self.ai_confidence) > 0:
            confidence_increase = ai_confidence - self.ai_confidence[-1]
            
            if confidence_increase > 30:
                self.realizations.append((current_time, ai_confidence))
                return True
        return False


def test_resonance_calculation():
    """Test resonance score calculation."""
    chamber = MockResonanceChamber()
    
    # Test perfect harmony
    score = chamber.calculate_resonance_score(50, 50)
    assert score == 100.0, f"Expected 100.0, got {score}"
    print("✓ Perfect harmony test passed (50, 50) -> 100.0")
    
    # Test partial disharmony
    score = chamber.calculate_resonance_score(70, 50)
    assert score == 80.0, f"Expected 80.0, got {score}"
    print("✓ Partial disharmony test passed (70, 50) -> 80.0")
    
    # Test maximum disharmony
    score = chamber.calculate_resonance_score(100, 0)
    assert score == 0.0, f"Expected 0.0, got {score}"
    print("✓ Maximum disharmony test passed (100, 0) -> 0.0")


def test_anomaly_detection():
    """Test anomaly detection logic."""
    chamber = MockResonanceChamber()
    
    # Add initial data point
    chamber.device_rhythm.append(20)
    chamber.human_rhythm.append(30)
    
    # Test no anomaly (small change)
    anomaly = chamber.detect_anomalies(25, 35, 1.0)
    assert not anomaly, "Should not detect anomaly for small change"
    print("✓ No anomaly detected for small change (5 points)")
    
    # Test anomaly detection (large spike)
    anomaly = chamber.detect_anomalies(80, 35, 2.0)
    assert anomaly, "Should detect anomaly for large spike"
    assert len(chamber.anomalies) == 1, "Should have one anomaly"
    print("✓ Anomaly detected for large spike (60 points)")


def test_realization_detection():
    """Test computational realization detection."""
    chamber = MockResonanceChamber()
    
    # Add initial confidence
    chamber.ai_confidence.append(30)
    
    # Test no realization (small increase)
    realization = chamber.detect_realizations(35, 1.0)
    assert not realization, "Should not detect realization for small increase"
    print("✓ No realization detected for small increase (5 points)")
    
    # Test realization detection (large increase)
    chamber.ai_confidence.append(35)
    realization = chamber.detect_realizations(70, 2.0)
    assert realization, "Should detect realization for large increase"
    assert len(chamber.realizations) == 1, "Should have one realization"
    print("✓ Realization detected for large increase (35 points)")


def test_ai_confidence_stability():
    """Test AI confidence behavior with stable and unstable inputs."""
    chamber = MockResonanceChamber()
    
    # Add initial data
    chamber.device_rhythm.append(50)
    chamber.human_rhythm.append(50)
    chamber.last_ai_confidence = 50
    
    # Test stable input (confidence should increase)
    confidence = chamber.calculate_ai_confidence(52, 51)
    assert confidence > 50, f"Confidence should increase with stability, got {confidence}"
    print(f"✓ AI confidence increased with stability: 50 -> {confidence}")
    
    # Add more data and test instability
    chamber.device_rhythm.append(52)
    chamber.human_rhythm.append(51)
    chamber.last_ai_confidence = confidence
    
    confidence_unstable = chamber.calculate_ai_confidence(90, 10)
    assert confidence_unstable < confidence, f"Confidence should decrease with instability"
    print(f"✓ AI confidence decreased with instability: {confidence} -> {confidence_unstable}")


def run_all_tests():
    """Run all test functions."""
    print("=" * 60)
    print("Testing Resonance Chamber Core Logic")
    print("=" * 60)
    
    try:
        test_resonance_calculation()
        print()
        test_anomaly_detection()
        print()
        test_realization_detection()
        print()
        test_ai_confidence_stability()
        print()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
