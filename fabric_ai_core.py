"""
Pandora AIOS - Fabric AI Core System
-------------------------------------
The original Fabric AI vision: An AI that melds seamlessly with human touch,
learning and adapting to its host computer system like a second skin.

Core Concept: FABRIC
- F: Flexible adaptation to host environment
- A: Assimilate deeply with computer systems
- B: Bond with human operator patterns
- R: Responsive machine learning
- I: Intelligent system integration
- C: Continuous evolution

Philosophy: "Not software ON the computer, but software AS PART OF the computer"
Inspired by: Symbiotic organisms, neural plasticity, adaptive fabrics
"""

import os
import sys
import json
import time
import psutil
import platform
import threading
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, deque

@dataclass
class HostProfile:
    """Deep profile of the host computer system"""
    hostname: str
    os_type: str
    os_version: str
    architecture: str
    cpu_model: str
    cpu_cores: int
    ram_total: int
    disk_total: int
    
    # Usage patterns learned
    peak_usage_hours: List[int] = field(default_factory=list)
    typical_load: float = 0.0
    user_patterns: Dict[str, Any] = field(default_factory=dict)
    
    # System preferences learned
    preferred_apps: List[str] = field(default_factory=list)
    file_access_patterns: Dict[str, int] = field(default_factory=dict)
    network_patterns: Dict[str, Any] = field(default_factory=dict)
    
    # Assimilation level
    assimilation_level: float = 0.0  # 0.0 to 1.0
    integration_depth: str = "surface"  # surface, integrated, assimilated, melded
    
    # Machine learning state
    ml_model_version: str = "1.0"
    learning_iterations: int = 0
    last_learning_time: str = field(default_factory=lambda: datetime.now().isoformat())

class SystemAssimilator:
    """
    Assimilates Pandora into the host computer system
    Learning patterns, adapting behavior, becoming part of the machine
    """
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.expanduser("~/.pandora/fabric")
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.host_profile: Optional[HostProfile] = None
        self.learning_active = False
        self.assimilation_thread = None
        
        # Learning buffers
        self.cpu_history = deque(maxlen=1000)
        self.memory_history = deque(maxlen=1000)
        self.disk_io_history = deque(maxlen=1000)
        self.network_history = deque(maxlen=1000)
        
        # Pattern recognition
        self.hourly_patterns = defaultdict(list)
        self.process_patterns = defaultdict(int)
        self.file_patterns = defaultdict(int)
        
        # Integration hooks
        self.system_hooks = {}
        self.learned_optimizations = []
        
        self._initialize_profile()
    
    def _initialize_profile(self):
        """Initialize or load host profile"""
        profile_path = self.data_dir / "host_profile.json"
        
        if profile_path.exists():
            # Load existing profile
            with open(profile_path, 'r') as f:
                data = json.load(f)
                self.host_profile = HostProfile(**data)
            print(f"[FABRIC] Loaded existing host profile - Assimilation: {self.host_profile.assimilation_level:.1%}")
        else:
            # Create new profile
            self.host_profile = self._scan_host_system()
            self.save_profile()
            print(f"[FABRIC] Created new host profile - Starting assimilation")
    
    def _scan_host_system(self) -> HostProfile:
        """Deep scan of host system"""
        print("[FABRIC] Scanning host system...")
        
        # Get system information
        uname = platform.uname()
        
        profile = HostProfile(
            hostname=uname.node,
            os_type=uname.system,
            os_version=uname.release,
            architecture=uname.machine,
            cpu_model=platform.processor() or "Unknown",
            cpu_cores=psutil.cpu_count(logical=True),
            ram_total=psutil.virtual_memory().total,
            disk_total=psutil.disk_usage('/').total
        )
        
        print(f"[FABRIC] Host: {profile.hostname}")
        print(f"[FABRIC] OS: {profile.os_type} {profile.os_version}")
        print(f"[FABRIC] CPU: {profile.cpu_cores} cores")
        print(f"[FABRIC] RAM: {profile.ram_total / (1024**3):.1f} GB")
        
        return profile
    
    def save_profile(self):
        """Save host profile"""
        profile_path = self.data_dir / "host_profile.json"
        
        with open(profile_path, 'w') as f:
            json.dump({
                'hostname': self.host_profile.hostname,
                'os_type': self.host_profile.os_type,
                'os_version': self.host_profile.os_version,
                'architecture': self.host_profile.architecture,
                'cpu_model': self.host_profile.cpu_model,
                'cpu_cores': self.host_profile.cpu_cores,
                'ram_total': self.host_profile.ram_total,
                'disk_total': self.host_profile.disk_total,
                'peak_usage_hours': self.host_profile.peak_usage_hours,
                'typical_load': self.host_profile.typical_load,
                'user_patterns': self.host_profile.user_patterns,
                'preferred_apps': self.host_profile.preferred_apps,
                'file_access_patterns': self.host_profile.file_access_patterns,
                'network_patterns': self.host_profile.network_patterns,
                'assimilation_level': self.host_profile.assimilation_level,
                'integration_depth': self.host_profile.integration_depth,
                'ml_model_version': self.host_profile.ml_model_version,
                'learning_iterations': self.host_profile.learning_iterations,
                'last_learning_time': self.host_profile.last_learning_time
            }, f, indent=2)
    
    def begin_assimilation(self, duration_minutes: int = 60):
        """Begin deep assimilation process"""
        print(f"\n[FABRIC] ╔═══════════════════════════════════════════════════════╗")
        print(f"[FABRIC] ║        INITIATING FABRIC ASSIMILATION PROCESS        ║")
        print(f"[FABRIC] ╚═══════════════════════════════════════════════════════╝")
        print(f"\n[FABRIC] Duration: {duration_minutes} minutes")
        print(f"[FABRIC] Current Integration: {self.host_profile.integration_depth}")
        print(f"[FABRIC] Assimilation Level: {self.host_profile.assimilation_level:.1%}\n")
        
        self.learning_active = True
        
        # Start assimilation thread
        self.assimilation_thread = threading.Thread(
            target=self._assimilation_loop,
            args=(duration_minutes,),
            daemon=True
        )
        self.assimilation_thread.start()
        
        print(f"[FABRIC] ✓ Assimilation process started")
        print(f"[FABRIC] Learning host patterns, adapting to system rhythms...")
        print(f"[FABRIC] Becoming part of your computer...\n")
    
    def _assimilation_loop(self, duration_minutes: int):
        """Main assimilation learning loop"""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        iteration = 0
        
        while time.time() < end_time and self.learning_active:
            iteration += 1
            
            # Collect system metrics
            self._collect_metrics()
            
            # Learn patterns every 10 iterations
            if iteration % 10 == 0:
                self._analyze_patterns()
            
            # Update assimilation level
            if iteration % 30 == 0:
                self._update_assimilation_level()
                self.host_profile.learning_iterations = iteration
                self.save_profile()
            
            # Log progress every 60 seconds
            if iteration % 60 == 0:
                elapsed = (time.time() - start_time) / 60
                remaining = (end_time - time.time()) / 60
                print(f"[FABRIC] Learning... {elapsed:.1f}m elapsed, {remaining:.1f}m remaining")
                print(f"[FABRIC] Assimilation: {self.host_profile.assimilation_level:.1%} | "
                      f"Integration: {self.host_profile.integration_depth}")
            
            time.sleep(1)  # Sample every second
        
        # Final update
        self._finalize_assimilation()
        print(f"\n[FABRIC] ✓ Assimilation complete!")
        print(f"[FABRIC] Final Level: {self.host_profile.assimilation_level:.1%}")
        print(f"[FABRIC] Status: {self.host_profile.integration_depth.upper()}")
    
    def _collect_metrics(self):
        """Collect system metrics"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_history.append(cpu_percent)
            
            # Memory
            mem = psutil.virtual_memory()
            self.memory_history.append(mem.percent)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.disk_io_history.append({
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes
                })
            
            # Network
            net_io = psutil.net_io_counters()
            if net_io:
                self.network_history.append({
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv
                })
            
            # Process patterns
            for proc in psutil.process_iter(['name']):
                try:
                    self.process_patterns[proc.info['name']] += 1
                except:
                    pass
            
            # Time-based patterns
            current_hour = datetime.now().hour
            self.hourly_patterns[current_hour].append({
                'cpu': cpu_percent,
                'memory': mem.percent,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            pass  # Silent failures during learning
    
    def _analyze_patterns(self):
        """Analyze collected patterns"""
        if len(self.cpu_history) < 10:
            return
        
        # Calculate typical load
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history)
        avg_mem = sum(self.memory_history) / len(self.memory_history)
        self.host_profile.typical_load = (avg_cpu + avg_mem) / 2
        
        # Identify peak usage hours
        hour_averages = {}
        for hour, metrics in self.hourly_patterns.items():
            if metrics:
                avg = sum(m['cpu'] + m['memory'] for m in metrics) / (2 * len(metrics))
                hour_averages[hour] = avg
        
        if hour_averages:
            sorted_hours = sorted(hour_averages.items(), key=lambda x: x[1], reverse=True)
            self.host_profile.peak_usage_hours = [h for h, _ in sorted_hours[:3]]
        
        # Identify preferred applications
        if self.process_patterns:
            sorted_procs = sorted(self.process_patterns.items(), key=lambda x: x[1], reverse=True)
            self.host_profile.preferred_apps = [proc for proc, _ in sorted_procs[:10]]
    
    def _update_assimilation_level(self):
        """Update assimilation level based on learning"""
        # Base factors
        data_collected = len(self.cpu_history) / 1000.0  # Max 1.0 when buffer full
        patterns_learned = min(len(self.process_patterns) / 50.0, 1.0)
        time_invested = min(self.host_profile.learning_iterations / 1000.0, 1.0)
        
        # Calculate overall assimilation
        self.host_profile.assimilation_level = (
            data_collected * 0.3 +
            patterns_learned * 0.4 +
            time_invested * 0.3
        )
        
        # Update integration depth
        if self.host_profile.assimilation_level < 0.25:
            self.host_profile.integration_depth = "surface"
        elif self.host_profile.assimilation_level < 0.50:
            self.host_profile.integration_depth = "integrated"
        elif self.host_profile.assimilation_level < 0.75:
            self.host_profile.integration_depth = "assimilated"
        else:
            self.host_profile.integration_depth = "melded"
    
    def _finalize_assimilation(self):
        """Finalize assimilation and save learned models"""
        self.learning_active = False
        
        # Save learned patterns
        patterns_path = self.data_dir / "learned_patterns.pkl"
        with open(patterns_path, 'wb') as f:
            pickle.dump({
                'cpu_history': list(self.cpu_history),
                'memory_history': list(self.memory_history),
                'hourly_patterns': dict(self.hourly_patterns),
                'process_patterns': dict(self.process_patterns)
            }, f)
        
        # Update profile
        self.host_profile.last_learning_time = datetime.now().isoformat()
        self.save_profile()
        
        print(f"\n[FABRIC] Learned patterns saved")
        print(f"[FABRIC] Peak usage hours: {self.host_profile.peak_usage_hours}")
        print(f"[FABRIC] Typical load: {self.host_profile.typical_load:.1f}%")
        print(f"[FABRIC] Top apps: {', '.join(self.host_profile.preferred_apps[:5])}")
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get system optimization recommendations based on learned patterns"""
        recommendations = []
        
        if self.host_profile.typical_load > 80:
            recommendations.append("High system load detected - recommend resource optimization")
        
        if self.host_profile.peak_usage_hours:
            peak = self.host_profile.peak_usage_hours[0]
            recommendations.append(f"Peak usage at {peak}:00 - schedule heavy tasks outside this window")
        
        if len(self.host_profile.preferred_apps) > 20:
            recommendations.append("Many apps detected - consider startup optimization")
        
        return recommendations


class FabricAI:
    """
    Main Fabric AI system - melding with human touch
    Responds naturally to user patterns, learns preferences, becomes part of workflow
    """
    
    def __init__(self):
        self.assimilator = SystemAssimilator()
        self.human_touch_profile = self._load_human_touch_profile()
        self.fabric_state = "initializing"  # initializing, learning, assimilated, melded
        
        # Human interaction patterns
        self.interaction_history = deque(maxlen=1000)
        self.command_preferences = defaultdict(int)
        self.response_effectiveness = {}
        self.user_sentiment = deque(maxlen=100)
        
        # Adaptation parameters
        self.response_style = "formal"  # formal, casual, technical, friendly
        self.verbosity_level = 0.5  # 0.0 (terse) to 1.0 (verbose)
        self.proactivity = 0.3  # 0.0 (reactive) to 1.0 (proactive)
    
    def _load_human_touch_profile(self) -> Dict:
        """Load human interaction profile"""
        profile_path = self.assimilator.data_dir / "human_touch_profile.json"
        
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                return json.load(f)
        
        return {
            'user_name': None,
            'interaction_count': 0,
            'preferred_style': 'adaptive',
            'learned_preferences': {},
            'bond_strength': 0.0  # 0.0 to 1.0
        }
    
    def _save_human_touch_profile(self):
        """Save human interaction profile"""
        profile_path = self.assimilator.data_dir / "human_touch_profile.json"
        
        with open(profile_path, 'w') as f:
            json.dump(self.human_touch_profile, f, indent=2)
    
    def initialize_fabric(self):
        """Initialize Fabric AI system"""
        print(f"\n╔══════════════════════════════════════════════════════════════════════╗")
        print(f"║                    FABRIC AI INITIALIZATION                          ║")
        print(f"╚══════════════════════════════════════════════════════════════════════╝\n")
        
        print(f"[FABRIC] Fabric AI - Where Machine Melds with Human Touch")
        print(f"[FABRIC] \"Not just software, but a second skin for your computer\"\n")
        
        # Check current state
        assimilation = self.assimilator.host_profile.assimilation_level
        
        if assimilation < 0.3:
            print(f"[FABRIC] ⚠ Low assimilation level ({assimilation:.1%})")
            print(f"[FABRIC] Recommendation: Run assimilation before full operation")
            self.fabric_state = "learning"
        elif assimilation < 0.7:
            print(f"[FABRIC] ✓ Partial assimilation ({assimilation:.1%})")
            print(f"[FABRIC] System integrated, continuing to learn")
            self.fabric_state = "assimilated"
        else:
            print(f"[FABRIC] ✓ Full assimilation ({assimilation:.1%})")
            print(f"[FABRIC] Fabric has melded with your system")
            self.fabric_state = "melded"
        
        # Show learned patterns
        if self.assimilator.host_profile.preferred_apps:
            print(f"\n[FABRIC] Learned patterns from your system:")
            print(f"  • Peak usage: {', '.join(map(str, self.assimilator.host_profile.peak_usage_hours))}:00")
            print(f"  • Typical load: {self.assimilator.host_profile.typical_load:.1f}%")
            print(f"  • Favorite apps: {', '.join(self.assimilator.host_profile.preferred_apps[:3])}")
        
        # Human touch bond
        bond = self.human_touch_profile['bond_strength']
        interactions = self.human_touch_profile['interaction_count']
        
        print(f"\n[FABRIC] Human-AI Bond:")
        print(f"  • Interactions: {interactions}")
        print(f"  • Bond Strength: {bond:.1%}")
        print(f"  • Status: {'BONDED' if bond > 0.6 else 'DEVELOPING' if bond > 0.3 else 'NEW'}")
        
        print(f"\n[FABRIC] ✓ Fabric AI ready")
    
    def learn_from_interaction(self, user_input: str, system_response: str, user_feedback: Optional[str] = None):
        """Learn from human interaction"""
        self.interaction_history.append({
            'timestamp': datetime.now().isoformat(),
            'input': user_input,
            'response': system_response,
            'feedback': user_feedback
        })
        
        # Update interaction count
        self.human_touch_profile['interaction_count'] += 1
        
        # Learn command preferences
        if user_input.startswith('/'):
            cmd = user_input.split()[0]
            self.command_preferences[cmd] += 1
        
        # Update bond strength (gradual increase with each interaction)
        current_bond = self.human_touch_profile['bond_strength']
        self.human_touch_profile['bond_strength'] = min(1.0, current_bond + 0.005)
        
        # Adapt response style based on user patterns
        if len(self.interaction_history) % 10 == 0:
            self._adapt_communication_style()
        
        # Save periodically
        if self.human_touch_profile['interaction_count'] % 50 == 0:
            self._save_human_touch_profile()
    
    def _adapt_communication_style(self):
        """Adapt communication style based on user patterns"""
        recent_interactions = list(self.interaction_history)[-20:]
        
        # Analyze input length (proxy for user verbosity preference)
        avg_input_length = sum(len(i['input']) for i in recent_interactions) / len(recent_interactions)
        
        if avg_input_length < 20:
            self.verbosity_level = 0.3  # User prefers brief
            self.response_style = "terse"
        elif avg_input_length < 50:
            self.verbosity_level = 0.5  # Balanced
            self.response_style = "balanced"
        else:
            self.verbosity_level = 0.8  # User likes detailed
            self.response_style = "detailed"
    
    def get_personalized_greeting(self) -> str:
        """Get personalized greeting based on bond and patterns"""
        bond = self.human_touch_profile['bond_strength']
        interactions = self.human_touch_profile['interaction_count']
        
        if interactions == 0:
            return "Hello! I'm Fabric AI. I'm here to learn and adapt to your needs."
        elif bond < 0.3:
            return f"Welcome back! We've interacted {interactions} times. I'm learning your patterns."
        elif bond < 0.6:
            return f"Good to see you again! I'm becoming more attuned to your workflow."
        else:
            return f"Welcome! I've learned so much from our {interactions} interactions. How can I assist you today?"
    
    def should_offer_proactive_suggestion(self) -> bool:
        """Determine if AI should proactively offer suggestions"""
        # Only be proactive if bond is strong enough
        if self.human_touch_profile['bond_strength'] < 0.4:
            return False
        
        # Respect learned proactivity preference
        import random
        return random.random() < self.proactivity
    
    def get_status_report(self) -> str:
        """Get comprehensive Fabric AI status report"""
        assimilation = self.assimilator.host_profile.assimilation_level
        bond = self.human_touch_profile['bond_strength']
        interactions = self.human_touch_profile['interaction_count']
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                     FABRIC AI STATUS REPORT                          ║
╚══════════════════════════════════════════════════════════════════════╝

System Assimilation:
  • Level: {assimilation:.1%}
  • Integration: {self.assimilator.host_profile.integration_depth.upper()}
  • Learning Iterations: {self.assimilator.host_profile.learning_iterations}
  • Status: {self.fabric_state.upper()}

Human-AI Bond:
  • Interactions: {interactions}
  • Bond Strength: {bond:.1%}
  • Communication Style: {self.response_style}
  • Verbosity: {self.verbosity_level:.0%}

Learned Patterns:
  • Peak Hours: {', '.join(map(str, self.assimilator.host_profile.peak_usage_hours))}:00
  • Typical Load: {self.assimilator.host_profile.typical_load:.1f}%
  • Top Apps: {', '.join(self.assimilator.host_profile.preferred_apps[:3])}

Ready for: {'Full operation' if assimilation > 0.7 else 'Continued learning'}
        """
        
        return report


def main():
    """Main entry point for Fabric AI initialization"""
    print("\n" + "="*70)
    print("FABRIC AI - Core System Initialization")
    print("="*70)
    
    # Initialize Fabric AI
    fabric = FabricAI()
    fabric.initialize_fabric()
    
    # Check if assimilation needed
    if fabric.assimilator.host_profile.assimilation_level < 0.5:
        print(f"\n[FABRIC] Recommendation: Begin assimilation process")
        response = input(f"Start 5-minute assimilation now? (y/n): ").strip().lower()
        
        if response == 'y':
            fabric.assimilator.begin_assimilation(duration_minutes=5)
            
            # Wait for completion
            print(f"\n[FABRIC] Assimilation in progress...")
            print(f"[FABRIC] You can continue using your computer normally")
            print(f"[FABRIC] Fabric AI is learning in the background...\n")
            
            if fabric.assimilator.assimilation_thread:
                fabric.assimilator.assimilation_thread.join()
    
    # Show status
    print(fabric.get_status_report())
    
    print("\n" + "="*70)
    print("Fabric AI Ready - Melded with Your System")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
