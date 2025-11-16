"""
Quick Start Example for Resonance Chamber

This script demonstrates how to use the Resonance Chamber tool
programmatically with custom settings.
"""

from resonance_chamber import ResonanceChamber


def example_basic():
    """Basic usage - default settings."""
    print("Starting Resonance Chamber with default settings...")
    chamber = ResonanceChamber()
    chamber.run()


def example_custom_history():
    """Custom usage - longer history window."""
    print("Starting Resonance Chamber with 100-point history...")
    # Keep 100 data points (approximately 50 seconds at 2 FPS)
    chamber = ResonanceChamber(history_length=100)
    chamber.run()


def example_short_history():
    """Custom usage - shorter history for quick responses."""
    print("Starting Resonance Chamber with 25-point history...")
    # Keep only 25 data points (approximately 12.5 seconds at 2 FPS)
    chamber = ResonanceChamber(history_length=25)
    chamber.run()


if __name__ == "__main__":
    # Choose which example to run
    example_basic()
    
    # Uncomment to try different examples:
    # example_custom_history()
    # example_short_history()
