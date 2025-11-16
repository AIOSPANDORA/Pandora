"""
Resonance Chamber - Diagnostic Tool for Pandora AI

This tool visualizes and tests the interaction between the user, device, and Pandora AI's
Fabric AI Core. It makes abstract concepts of "Resonance" and "Relativism" tangible and testable.

The dashboard displays real-time metrics:
- Device Rhythm: CPU utilization normalized to 0-100
- Human Rhythm: Keyboard and mouse activity frequency
- AI Predictive Confidence: Simulated AI state metric
- Resonance Score: Harmony between user and device (100 - abs(Device - Human))

Anomalies (red X) and Computational Realizations (green O) are marked on the timeline.
"""

import time
import threading
from collections import deque
import psutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pynput import keyboard, mouse


class ResonanceChamber:
    """
    Main class encapsulating the Resonance Chamber diagnostic tool.
    
    This class manages data collection, processing, and visualization of the
    resonance between human input, device state, and AI predictions.
    """
    
    def __init__(self, history_length=50):
        """
        Initialize the Resonance Chamber.
        
        Args:
            history_length: Number of data points to keep in history for plotting
        """
        self.history_length = history_length
        
        # Data storage - using deque for efficient append/pop operations
        self.timestamps = deque(maxlen=history_length)
        self.device_rhythm = deque(maxlen=history_length)
        self.human_rhythm = deque(maxlen=history_length)
        self.ai_confidence = deque(maxlen=history_length)
        self.resonance_score = deque(maxlen=history_length)
        
        # Anomaly and realization markers
        self.anomalies = []  # List of (time, value) tuples for red X markers
        self.realizations = []  # List of (time, value) tuples for green O markers
        
        # Human activity tracking
        self.keyboard_events = 0
        self.mouse_events = 0
        self.last_human_rhythm = 0
        self.last_ai_confidence = 50.0  # Start at neutral
        self.activity_lock = threading.Lock()
        
        # Time tracking
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        # Listener flags
        self.running = True
        self.keyboard_listener = None
        self.mouse_listener = None
        
    def on_keyboard_press(self, key):
        """Callback for keyboard press events."""
        with self.activity_lock:
            self.keyboard_events += 1
    
    def on_mouse_move(self, x, y):
        """Callback for mouse movement events."""
        with self.activity_lock:
            self.mouse_events += 1
    
    def on_mouse_click(self, x, y, button, pressed):
        """Callback for mouse click events."""
        if pressed:
            with self.activity_lock:
                self.mouse_events += 2  # Clicks are more significant than moves
    
    def start_input_listeners(self):
        """Start the pynput listeners in separate threads."""
        # Start keyboard listener
        self.keyboard_listener = keyboard.Listener(on_press=self.on_keyboard_press)
        self.keyboard_listener.daemon = True
        self.keyboard_listener.start()
        
        # Start mouse listener
        self.mouse_listener = mouse.Listener(
            on_move=self.on_mouse_move,
            on_click=self.on_mouse_click
        )
        self.mouse_listener.daemon = True
        self.mouse_listener.start()
    
    def stop_input_listeners(self):
        """Stop the input listeners."""
        self.running = False
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        if self.mouse_listener:
            self.mouse_listener.stop()
    
    def get_device_rhythm(self):
        """
        Calculate Device Rhythm based on CPU utilization.
        
        Returns:
            float: CPU utilization normalized to 0-100 scale
        """
        # Get CPU percent (already normalized to 0-100)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        return cpu_percent
    
    def get_human_rhythm(self):
        """
        Calculate Human Rhythm based on keyboard and mouse activity.
        
        The rhythm score is calculated from the frequency of input events,
        scaled to a 0-100 range. More activity = higher rhythm.
        
        Returns:
            float: Human activity score normalized to 0-100 scale
        """
        with self.activity_lock:
            # Get total events since last update
            total_events = self.keyboard_events + self.mouse_events
            # Reset counters
            self.keyboard_events = 0
            self.mouse_events = 0
        
        # Calculate events per second, scale to 0-100
        # Assume 20 events/sec = 100 rhythm (high activity)
        events_per_sec = total_events / 0.5  # Update interval is ~0.5 seconds
        rhythm = min(100.0, (events_per_sec / 20.0) * 100.0)
        
        return rhythm
    
    def calculate_ai_confidence(self, device_rhythm, human_rhythm):
        """
        Simulate AI Predictive Confidence metric.
        
        The confidence increases when device and human rhythms are stable,
        and decreases during periods of high activity or rapid change.
        
        Args:
            device_rhythm: Current device rhythm value
            human_rhythm: Current human rhythm value
            
        Returns:
            float: AI confidence score (0-100)
        """
        # Calculate stability: lower values mean more stable
        if len(self.device_rhythm) > 1:
            device_change = abs(device_rhythm - self.device_rhythm[-1])
            human_change = abs(human_rhythm - self.human_rhythm[-1])
        else:
            device_change = 0
            human_change = 0
        
        # Total change (instability measure)
        total_change = device_change + human_change
        
        # Update confidence: decrease with high change, increase with stability
        if total_change > 30:
            # High instability - confidence drops
            confidence_delta = -5
        elif total_change < 10:
            # High stability - confidence increases
            confidence_delta = 3
        else:
            # Moderate change - slight decrease
            confidence_delta = -1
        
        # Update confidence with bounds checking
        new_confidence = max(0.0, min(100.0, self.last_ai_confidence + confidence_delta))
        
        return new_confidence
    
    def calculate_resonance_score(self, device_rhythm, human_rhythm):
        """
        Calculate Resonance Score: harmony between user and device.
        
        Formula: 100 - abs(Device_Rhythm - Human_Rhythm)
        Perfect harmony (same values) = 100
        Maximum disharmony (opposite extremes) = 0
        
        Args:
            device_rhythm: Current device rhythm value
            human_rhythm: Current human rhythm value
            
        Returns:
            float: Resonance score (0-100)
        """
        return 100.0 - abs(device_rhythm - human_rhythm)
    
    def detect_anomalies(self, device_rhythm, human_rhythm, current_time):
        """
        Detect anomalies: spikes > 50 points in a single time step.
        
        Args:
            device_rhythm: Current device rhythm value
            human_rhythm: Current human rhythm value
            current_time: Current timestamp
        """
        if len(self.device_rhythm) > 0:
            device_spike = abs(device_rhythm - self.device_rhythm[-1])
            human_spike = abs(human_rhythm - self.human_rhythm[-1])
            
            # Check for significant spikes
            if device_spike > 50 or human_spike > 50:
                # Mark anomaly at current time
                max_spike = max(device_spike, human_spike)
                self.anomalies.append((current_time, max_spike))
                # Keep only recent anomalies (last history_length)
                if len(self.anomalies) > self.history_length:
                    self.anomalies.pop(0)
    
    def detect_realizations(self, ai_confidence, current_time):
        """
        Detect Computational Realizations: AI confidence increases > 30 points.
        
        This simulates the AI having an "aha!" moment and understanding a new pattern.
        
        Args:
            ai_confidence: Current AI confidence value
            current_time: Current timestamp
        """
        if len(self.ai_confidence) > 0:
            confidence_increase = ai_confidence - self.ai_confidence[-1]
            
            # Check for significant confidence increase
            if confidence_increase > 30:
                self.realizations.append((current_time, ai_confidence))
                # Keep only recent realizations
                if len(self.realizations) > self.history_length:
                    self.realizations.pop(0)
    
    def update_data(self):
        """
        Update all metrics and collect new data point.
        
        This method is called periodically to gather new measurements.
        """
        current_time = time.time() - self.start_time
        
        # Collect metrics
        device_rhythm = self.get_device_rhythm()
        human_rhythm = self.get_human_rhythm()
        
        # Detect anomalies before updating data
        self.detect_anomalies(device_rhythm, human_rhythm, current_time)
        
        # Calculate derived metrics
        ai_confidence = self.calculate_ai_confidence(device_rhythm, human_rhythm)
        
        # Detect realizations
        self.detect_realizations(ai_confidence, current_time)
        
        resonance = self.calculate_resonance_score(device_rhythm, human_rhythm)
        
        # Store data
        self.timestamps.append(current_time)
        self.device_rhythm.append(device_rhythm)
        self.human_rhythm.append(human_rhythm)
        self.ai_confidence.append(ai_confidence)
        self.resonance_score.append(resonance)
        
        # Update last values for next iteration
        self.last_human_rhythm = human_rhythm
        self.last_ai_confidence = ai_confidence
    
    def init_plot(self):
        """Initialize the plot figure and subplots."""
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 9))
        self.fig.suptitle('Resonance Chamber - Pandora AI Diagnostic Tool', 
                         fontsize=14, fontweight='bold')
        
        # Configure subplots
        self.axes[0].set_ylabel('Rhythm Score', fontsize=10)
        self.axes[0].set_title('Device Rhythm vs. Human Rhythm', fontsize=11)
        self.axes[0].grid(True, alpha=0.3)
        self.axes[0].set_ylim(-5, 105)
        
        self.axes[1].set_ylabel('Confidence Score', fontsize=10)
        self.axes[1].set_title('AI Predictive Confidence', fontsize=11)
        self.axes[1].grid(True, alpha=0.3)
        self.axes[1].set_ylim(-5, 105)
        
        self.axes[2].set_xlabel('Time (seconds)', fontsize=10)
        self.axes[2].set_ylabel('Resonance Score', fontsize=10)
        self.axes[2].set_title('Overall Resonance Score', fontsize=11)
        self.axes[2].grid(True, alpha=0.3)
        self.axes[2].set_ylim(-5, 105)
        
        plt.tight_layout()
        
        return self.fig,
    
    def animate(self, frame):
        """
        Animation update function called by FuncAnimation.
        
        Args:
            frame: Frame number (provided by FuncAnimation)
        """
        # Update data
        self.update_data()
        
        # Clear all axes
        for ax in self.axes:
            ax.clear()
        
        # Convert deques to lists for plotting
        times = list(self.timestamps)
        
        # Plot 1: Device Rhythm vs. Human Rhythm
        if len(times) > 0:
            self.axes[0].plot(times, list(self.device_rhythm), 
                            label='Device Rhythm', color='blue', linewidth=2)
            self.axes[0].plot(times, list(self.human_rhythm), 
                            label='Human Rhythm', color='orange', linewidth=2)
            
            # Mark anomalies on this plot
            for anomaly_time, _ in self.anomalies:
                if anomaly_time in times:
                    idx = times.index(anomaly_time)
                    if idx < len(self.device_rhythm):
                        y_val = max(self.device_rhythm[idx], self.human_rhythm[idx])
                        self.axes[0].plot(anomaly_time, y_val, 'rx', 
                                        markersize=12, markeredgewidth=3, 
                                        label='Anomaly' if anomaly_time == self.anomalies[0][0] else '')
            
            self.axes[0].legend(loc='upper left', fontsize=8)
        
        self.axes[0].set_ylabel('Rhythm Score', fontsize=10)
        self.axes[0].set_title('Device Rhythm vs. Human Rhythm', fontsize=11)
        self.axes[0].grid(True, alpha=0.3)
        self.axes[0].set_ylim(-5, 105)
        
        # Plot 2: AI Predictive Confidence
        if len(times) > 0:
            self.axes[1].plot(times, list(self.ai_confidence), 
                            label='AI Confidence', color='green', linewidth=2)
            
            # Mark realizations on this plot
            for realization_time, confidence_val in self.realizations:
                self.axes[1].plot(realization_time, confidence_val, 'go', 
                                markersize=10, markeredgewidth=2, 
                                label='Realization' if realization_time == self.realizations[0][0] else '')
            
            self.axes[1].legend(loc='upper left', fontsize=8)
        
        self.axes[1].set_ylabel('Confidence Score', fontsize=10)
        self.axes[1].set_title('AI Predictive Confidence', fontsize=11)
        self.axes[1].grid(True, alpha=0.3)
        self.axes[1].set_ylim(-5, 105)
        
        # Plot 3: Overall Resonance Score
        if len(times) > 0:
            self.axes[2].plot(times, list(self.resonance_score), 
                            label='Resonance Score', color='purple', linewidth=2)
            self.axes[2].legend(loc='upper left', fontsize=8)
        
        self.axes[2].set_xlabel('Time (seconds)', fontsize=10)
        self.axes[2].set_ylabel('Resonance Score', fontsize=10)
        self.axes[2].set_title('Overall Resonance Score', fontsize=11)
        self.axes[2].grid(True, alpha=0.3)
        self.axes[2].set_ylim(-5, 105)
        
        plt.tight_layout()
        
        return self.axes
    
    def run(self):
        """
        Start the Resonance Chamber dashboard.
        
        This method initializes input listeners, sets up the matplotlib animation,
        and displays the live dashboard.
        """
        print("Starting Resonance Chamber...")
        print("Monitoring device and human activity...")
        print("Press Ctrl+C or close the window to stop.")
        
        # Start input monitoring
        self.start_input_listeners()
        
        # Create animation
        # Update interval: 500ms (2 FPS) for smooth but not too frequent updates
        anim = animation.FuncAnimation(
            self.fig, 
            self.animate, 
            init_func=self.init_plot,
            interval=500,  # milliseconds
            blit=False,
            cache_frame_data=False
        )
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nStopping Resonance Chamber...")
        finally:
            self.stop_input_listeners()
            plt.close()
        
        print("Resonance Chamber stopped.")


if __name__ == "__main__":
    """
    Main entry point for the Resonance Chamber diagnostic tool.
    
    Creates and runs a ResonanceChamber instance with default settings.
    """
    # Create chamber with 50 data points of history
    chamber = ResonanceChamber(history_length=50)
    
    # Run the dashboard
    chamber.run()
