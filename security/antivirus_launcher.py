"""
Pandora Antivirus Launcher
-------------------------
This script integrates ClamAV (open-source antivirus engine) with Pandora AIOS for real-time and on-demand malware scanning.

Features:
- Checks if ClamAV daemon is running (starts it if possible).
- Can scan single files or entire directories.
- Logs detections and actions.
- Easily callable from other Pandora scripts/modules.

Requirements:
- Linux host (Debian/Ubuntu recommended)
- ClamAV installed (`sudo apt-get install clamav clamav-daemon`)
- `pyclamd` python package (`pip install pyclamd`)
"""

import os
import sys
import time
import logging

try:
    import pyclamd
except ImportError:
    pyclamd = None

ANTIVIRUS_LOGFILE = "/var/log/pandora_antivirus.log"

def setup_logging():
    logging.basicConfig(
        filename=ANTIVIRUS_LOGFILE,
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

def is_clamd_running():
    # Try connecting with pyclamd or check systemd status
    if not pyclamd:
        return False
    try:
        cd = pyclamd.ClamdAgnostic()
        return cd.ping()
    except Exception:
        return False

def try_start_clamd():
    print("Attempting to start clamd service...")
    try:
        os.system("sudo systemctl start clamav-daemon || sudo service clamav-daemon start")
        time.sleep(3)
    except Exception:
        pass

def scan_path(target_path):
    if not pyclamd:
        logging.error("pyclamd not installed. Cannot scan files.")
        return
    try:
        cd = pyclamd.ClamdAgnostic()
        if not cd.ping():
            print("ClamAV daemon not responding.")
            logging.error("ClamAV daemon not responding.")
            return
        
        if os.path.isfile(target_path):
            result = cd.scan_file(target_path)
        elif os.path.isdir(target_path):
            result = cd.scan_path(target_path)
        else:
            logging.warning(f"Path {target_path} does not exist.")
            return
        if not result:
            print(f"{target_path}: CLEAN")
            logging.info(f"{target_path}: CLEAN")
        else:
            print(f"!! THREAT DETECTED !! {result}")
            logging.warning(f"THREAT DETECTED: {result}")
    except Exception as e:
        logging.error(f"Error scanning {target_path}: {e}")

def interactive_scan():
    print("Pandora Antivirus Launcher")
    print("-------------------------")
    while True:
        target = input("Enter file or directory to scan (or 'exit'): ").strip()
        if target.lower() in ('exit', 'quit'):
            break
        scan_path(target)

def trigger_realtime_scan_on_upload(file_path):
    # This is a hook for Pandora uploads: call after saving any user-uploaded file
    scan_path(file_path)

if __name__ == "__main__":
    setup_logging()
    if not pyclamd:
        print("pyclamd not installed. Please run: pip install pyclamd")
        sys.exit(1)
    if not is_clamd_running():
        print("ClamAV daemon not running. Trying to start it...")
        try_start_clamd()
        if not is_clamd_running():
            print("Failed to start ClamAV. Please install/start clamav-daemon.")
            sys.exit(1)
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            scan_path(arg)
    else:
        interactive_scan()