import subprocess
import time
import os
import sys

# Core: Only merge code/branches if all key Pandora systems are up and healthy after install/boot

def all_systems_healthy():
    # Core logs to check. Adjust paths/names as your stack needs.
    logs = [
        '/var/log/pandora_supervisor.log',
        '/var/log/pandora_quantum_firewall.log',
        '/var/log/pandora_avfirewall.log'
    ]
    for log in logs:
        if not os.path.exists(log):
            print(f"Missing health log: {log}")
            return False
        with open(log, "r") as f:
            content = f.read()
            if any(err in content.lower() for err in ("critical", "fail", "panic", "error")):
                print(f"{log} reports error keyword, merge will not run.")
                return False
    return True

def merge_to_production_branch(repo_path, merge_from='feature/autoboost', target='main'):
    repo_path = os.path.expanduser(repo_path)
    cmds = [
        f"cd {repo_path}",
        "git fetch",
        f"git checkout {target}",
        "git pull",
        f"git merge {merge_from} -m 'Pandora: automatic security/quantum merge after verified boot.'",
        "git push"
    ]
    result = subprocess.run(" && ".join(cmds), shell=True)
    return result.returncode == 0

def run_boot_merge():
    pandora_path = "~/Pandora"  # Adjust if needed
    delay = 15  # Let logs be written and services start
    print("[Pandora Merge] Waiting for core system boot logs...")
    time.sleep(delay)
    if all_systems_healthy():
        print("[Pandora Merge] All health checks passed. Merging upgrades to main branch.")
        if merge_to_production_branch(pandora_path):
            print("[Pandora Merge] Merge and push to production complete.")
            sys.exit(0)
        else:
            print("[Pandora Merge] Git merge failed (manual check required).")
            sys.exit(2)
    else:
        print("[Pandora Merge] Health verification failed. Skipping upgrade merge for safety.")
        sys.exit(1)

if __name__ == "__main__":
    run_boot_merge()