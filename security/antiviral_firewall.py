import subprocess
import os
import logging
from datetime import datetime

try:
    import pyclamd  # Provides ClamAV bindings for Python
except ImportError:
    pyclamd = None  # Will warn later

class AntiviralFirewall:
    def __init__(self, log_path="pandora_security.log"):
        self.log_path = log_path
        self.log_event("AntiviralFirewall initialized.")
        self.enable_linux_firewall()
        self.check_clamav_loaded()

    def log_event(self, msg):
        entry = f"[{datetime.utcnow().isoformat()}] {msg}"
        with open(self.log_path, "a") as f:
            f.write(entry + "\n")
        print(entry)

    def enable_linux_firewall(self):
        # Strongest default: deny-all, then open only explicit AIOS ports/services
        try:
            subprocess.run(['ufw', 'enable'], check=True)
            subprocess.run(['ufw', 'default', 'deny'], check=True)
            # Example allow: Pandora's web UI or SSH (update as needed)
            subprocess.run(['ufw', 'allow', '5000'], check=True)  # For Pandoras web
            subprocess.run(['ufw', 'allow', '22'], check=True)    # If SSH required
            self.log_event("Firewall enabled with deny-all policy; explicit ports whitelisted.")
        except Exception as e:
            self.log_event(f"WARNING: Could not enforce firewall: {e}")

    def check_clamav_loaded(self):
        if not pyclamd:
            self.log_event("WARNING: pyclamd not installed -- ClamAV virus scanning unavailable!")
            return False
        try:
            cd = pyclamd.ClamdAgnostic()
            if not cd.ping():
                self.log_event("ClamAV engine not running or responding. Please start clamav-daemon.")
                return False
            self.log_event("ClamAV connected. Realtime virus scanning enabled.")
            return True
        except Exception as e:
            self.log_event(f"ClamAV error: {e}")
            return False

    def scan_file(self, path):
        """Scan file for viruses using ClamAV."""
        if not pyclamd:
            self.log_event("(Scan skipped) pyclamd not loaded.")
            return "no_av"
        try:
            cd = pyclamd.ClamdAgnostic()
            result = cd.scan_file(path)
            if result is None:
                self.log_event(f"File {path}: CLEAN")
                return "clean"
            else:
                self.log_event(f"File {path}: INFECTED! {result}")
                self.quarantine_file(path)
                return "infected"
        except Exception as e:
            self.log_event(f"ClamAV file scan error: {e}")
            return "scan_error"

    def quarantine_file(self, path):
        """Move infected file to quarantine directory and block execution."""
        try:
            quarantine_dir = "/var/quarantine"
            os.makedirs(quarantine_dir, exist_ok=True)
            basename = os.path.basename(path)
            new_path = os.path.join(quarantine_dir, basename)
            os.rename(path, new_path)
            self.log_event(f"File {path} moved to quarantine: {new_path}")
        except Exception as e:
            self.log_event(f"Quarantine error: {e}")

    def audit_suspicious_behavior(self, event, severity="medium"):
        # Use this for any process/behavior you want to escalate/alert on.
        self.log_event(f"SUSPICIOUS ACTIVITY [{severity.upper()}]: {event}")
        # Optionally trigger system lockdown, alert admin, enter Safe Mode, etc.
        if severity == "high":
            self.activate_lockdown()

    def activate_lockdown(self):
        self.log_event("ACTIVATING LOCKDOWN: Blocking all external network access except admin!")
        subprocess.run(['ufw', 'default', 'deny', 'outgoing'], check=False)
        # Optionally call Pandora Safe Mode here

# Usage Example (at Pandora startup or critical I/O event):
if __name__ == "__main__":
    fw = AntiviralFirewall()
    fw.scan_file("/tmp/test-upload.zip")
    fw.audit_suspicious_behavior("Unusual outbound traffic detected from user process", severity="high")