# Resonance Chamber - Diagnostic Tool

## Overview

The Resonance Chamber is a diagnostic tool that visualizes and tests the interaction between the user, the device, and the Pandora AI's Fabric AI Core. It makes abstract concepts of "Resonance" and "Relativism" tangible and testable through real-time metrics and visualizations.

## Features

### Real-time Metrics

1. **Device Rhythm**: Real-time CPU utilization normalized to 0-100 scale (via psutil)
2. **Human Rhythm**: Keyboard and mouse activity frequency (via pynput)
3. **AI Predictive Confidence**: Simulated AI state metric that increases with stability
4. **Resonance Score**: Harmony between user and device (100 - abs(Device - Human))

### Visualizations

The dashboard includes three real-time updating subplots:

1. **Device Rhythm vs. Human Rhythm**: Shows both rhythms on the same axes for comparison
2. **AI Predictive Confidence**: Displays the simulated AI's confidence over time
3. **Overall Resonance Score**: Shows the harmony metric between user and device

### Anomaly Detection

- **Red 'X' markers**: Appear when Device or Human Rhythm spikes by more than 50 points in a single time step
- Helps identify sudden changes in system or user behavior

### Computational Realizations

- **Green 'O' markers**: Appear when AI Predictive Confidence increases by more than 30 points
- Simulates the AI having an "aha!" moment and understanding a new pattern

## Installation

Install the required dependencies:

```bash
pip install -r requirements-dev.txt
```

Or install individually:

```bash
pip install matplotlib psutil pynput
```

## Usage

### Running the Tool

Simply execute the script:

```bash
python resonance_chamber.py
```

The dashboard will open in a new window and begin monitoring:
- Device CPU utilization
- Your keyboard and mouse activity
- Simulated AI confidence
- Overall resonance between you and the device

### Interpreting the Dashboard

- **High Resonance Score (>80)**: Good harmony between user activity and device state
- **Low Resonance Score (<50)**: Disharmony - user and device are out of sync
- **Rising AI Confidence**: Stable patterns detected
- **Falling AI Confidence**: Unstable or rapidly changing patterns
- **Red X markers**: Sudden spikes indicating anomalies
- **Green O markers**: AI "realizations" - significant pattern understanding

### Stopping the Tool

- Press `Ctrl+C` in the terminal, or
- Close the matplotlib window

## Code Structure

### Main Class: `ResonanceChamber`

The tool is built around a single class that encapsulates all functionality:

- **Data Collection**: Methods for gathering device and human activity metrics
- **Metric Calculation**: Algorithms for computing derived metrics (resonance, AI confidence)
- **Anomaly Detection**: Logic for identifying significant events
- **Visualization**: Matplotlib-based real-time dashboard
- **Threading**: Separate thread for input monitoring to avoid blocking the UI

### Key Methods

- `get_device_rhythm()`: Collects CPU utilization
- `get_human_rhythm()`: Calculates activity from input events
- `calculate_ai_confidence()`: Simulates AI state based on stability
- `calculate_resonance_score()`: Computes harmony metric
- `detect_anomalies()`: Identifies significant spikes
- `detect_realizations()`: Identifies AI "aha" moments
- `animate()`: Updates the dashboard plots in real-time

## Testing

A test suite is included to validate the core logic:

```bash
python test_resonance_chamber.py
```

Tests cover:
- Resonance score calculation
- Anomaly detection logic
- Realization detection logic
- AI confidence behavior with stable/unstable inputs

## Technical Details

### Threading Model

The tool uses Python's `threading` module to run the pynput listeners in separate threads. This ensures that input monitoring doesn't block the matplotlib UI thread.

### Update Frequency

- Dashboard updates: 500ms (2 FPS)
- CPU sampling: 100ms intervals
- Input events: Captured in real-time, aggregated every 500ms

### Data History

By default, the tool maintains 50 data points of history (approximately 25 seconds at 2 FPS). This can be adjusted via the `history_length` parameter.

## Philosophy

The Resonance Chamber embodies Pandora's core principles:

1. **Harmony**: Measuring the synchronization between human and machine
2. **Awareness**: Real-time feedback on the state of the system
3. **Adaptation**: AI confidence that responds to patterns and stability
4. **Transparency**: Visual representation of abstract concepts

## Limitations

- AI Predictive Confidence is currently simulated; integration with actual AI core is planned
- Human Rhythm calculation is simplified; more sophisticated activity analysis could be implemented
- Requires graphical environment (X server) for the dashboard

## Future Enhancements

- Integration with actual Pandora AI Fabric Core
- Historical data export and analysis
- Configurable thresholds for anomaly/realization detection
- Multiple device metric sources (memory, network, disk I/O)
- Advanced human activity patterns (typing speed, mouse acceleration)
- Network-based multi-device resonance tracking
