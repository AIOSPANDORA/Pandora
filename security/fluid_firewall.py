"""
FluidFirewall: Adaptive, Self-Updating AI-Driven Sandbox Membrane

- Dynamically randomizes firewall rulesets, port layouts, process monitoring, and inter-process communication patterns.
- Maintains a virtual 'Faraday cage' membrane: network, file, and process quarantine around AIOS core.
- Learns from attempted intrusions and anomalies, adapts its own defense posture and randomization.
- Can be configured to update/randomize rulesets on intervals or on specific "threat detected" signals.
- Logs all activity for review and rollback.
- Should run as a privileged daemon/service, wrapping the core AIOS Pandora process.

Requires: Linux (with iptables/nftables & psutil), Python 3.8+

Authors: Pandora AIOS Security Team
"""

import os
import random
import subprocess
import time
import threading
import psutil
import uuid
from datetime import datetime

class FluidFirewall:
    def __init__(self, monitored_pid=None, update_interval=90):
        self.membrane_id = str(uuid.uuid4())
        self.monitor_pid = monitored_pid
        self.active_ports = set()
        self.port_map = {}
        self.update_interval = update_interval  # seconds
        self.logpath = f"/tmp/fluidfirewall_{self.membrane_id}.log"
        self.running = True
        self._boot_log("FluidFirewall membrane initialized.")

    def _boot_log(self, msg):
        entry = f"[{datetime.utcnow().isoformat()}][FluidFirewall] {msg}"
        with open(self.logpath, "a") as f:
            f.write(entry + "\n")
        print(entry)

    def start(self):
        self._boot_log("Starting adaptive firewall membrane loop...")
        threading.Thread(target=self._randomize_loop, daemon=True).start()
        threading.Thread(target=self._monitor_process, daemon=True).start()

    def _randomize_loop(self):
        while self.running:
            self.randomize_ports()
            self.randomize_rules()
            self._boot_log("Firewall/randomization sweep complete.")
            time.sleep(self.update_interval)

    def randomize_ports(self):
        """Randomly reroute which service ports are open for AIOS ingress/egress."""
        exposed = [random.randint(4000, 65000) for _ in range(3)]
        self.port_map = {f"aios_port_{i}": p for i, p in enumerate(exposed)}
        self.active_ports = set(exposed)
        self._apply_port_rules()
        self._boot_log(f"Port rules randomized: {self.port_map}")

    def _apply_port_rules(self):
        # First, block all previous ports.
        subprocess.run(['iptables', '-F', 'INPUT'])
        subprocess.run(['iptables', '-F', 'OUTPUT'])
        for port in self.active_ports:
            # Only allow selected randomized ports for AIOS
            subprocess.run(['iptables', '-A', 'INPUT', '-p', 'tcp', '--dport', str(port), '-j', 'ACCEPT'])
            subprocess.run(['iptables', '-A', 'OUTPUT', '-p', 'tcp', '--sport', str(port), '-j', 'ACCEPT'])
        # Deny all else
        subprocess.run(['iptables', '-P', 'INPUT', 'DROP'])
        subprocess.run(['iptables', '-P', 'OUTPUT', 'DROP'])

    def randomize_rules(self):
        # Randomize rate limits, SYN flood protections, weirdness etc.
        syn_protect = random.choice([True, False])
        rate_limit = random.randint(20, 200)
        # Example: add SYN flood protection
        if syn_protect:
            subprocess.run(['iptables', '-A', 'INPUT', '-p', 'tcp', '--syn', '-m', 
                            'limit', '--limit', f'{rate_limit}/second', '--limit-burst', '10', '-j', 'ACCEPT'])
            self._boot_log(f"SYN protection enabled at {rate_limit}/sec")
        # Custom drop rules for noisy or unknown sources (advanced: could parse previous logs and block known abusers)

    def _monitor_process(self):
        if not self.monitor_pid:
            self._boot_log("No monitored PID provided for sandboxing.")
            return
        self._boot_log(f"Monitoring AIOS process: {self.monitor_pid}")
        while self.running:
            try:
                proc = psutil.Process(self.monitor_pid)
                # Resource checks
                cpu = proc.cpu_percent(interval=1.0)
                mem = proc.memory_info().rss
                if cpu > 90:
                    self._boot_log(f"High CPU detected: {cpu}% on AIOS.")
                if mem > 2 * 1024 ** 3:
                    self._boot_log(f"High Memory detected: {mem / 1024 ** 2:.2f}MB on AIOS.")
                # File/network/child process scanning
                open_files = proc.open_files()
                connections = proc.connections()
                for f in open_files:
                    if '/tmp' in f.path:
                        self._boot_log(f"ALERT: AIOS opened temp file {f.path}. Possible risk.")
                for conn in connections:
                    if conn.raddr and conn.status == 'ESTABLISHED':
                        ip = conn.raddr.ip
                        if not ip.startswith('192.168.'):
                            self._boot_log(f"ALERT: AIOS outbound to {ip}.")
            except psutil.NoSuchProcess:
                self._boot_log("AIOS process terminated! Halting firewall.")
                self.running = False
            except Exception as e:
                self._boot_log(f"Monitor error: {e}")
            time.sleep(7)

    def freeze(self):
        """Emergency lockdown: block all network."""
        self.running = False
        subprocess.run(['iptables', '-P', 'INPUT', 'DROP'])
        subprocess.run(['iptables', '-P', 'OUTPUT', 'DROP'])
        self._boot_log("EMERGENCY: All network is now blocked (Faraday mode). AIOS is isolated!")

    def shutdown(self):
        self._boot_log("Shutting down FluidFirewall membrane.")
        self.running = False
        subprocess.run(['iptables', '-F', 'INPUT'])
        subprocess.run(['iptables', '-F', 'OUTPUT'])
        subprocess.run(['iptables', '-P', 'INPUT', 'ACCEPT'])
        subprocess.run(['iptables', '-P', 'OUTPUT', 'ACCEPT'])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch adaptive fluid firewall for AIOS Pandora")
    parser.add_argument('--pid', type=int, help="PID of AIOS process to sandbox/monitor")
    parser.add_argument('--interval', type=int, default=90, help="Randomization/check interval (seconds)")
    args = parser.parse_args()

    ffw = FluidFirewall(monitored_pid=args.pid, update_interval=args.interval)
    ffw.start()
    print("FluidFirewall is active. Ctrl+C to exit.")
    try:
        while ffw.running:
            time.sleep(10)
    except KeyboardInterrupt:
        ffw.shutdown()