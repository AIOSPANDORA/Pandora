import psutil, sys, time

def healthy():
    # Example: check for low RAM, high CPU, etc.
    mem = psutil.virtual_memory()
    if mem.available < 512 * 1024 * 1024:  # <512MB free
        print("[ERROR] Low RAM!")
        return False
    cpu = psutil.cpu_percent(interval=1)
    if cpu > 95:
        print("[ERROR] CPU overloaded!")
        return False
    # (Add NVIDIA/Intel/GPU checks here as needed)
    return True

if __name__ == "__main__":
    if "--boot" in sys.argv:
        print("Running health monitor on boot...")
        for _ in range(5):
            if not healthy():
                sys.exit(1)
            time.sleep(1)
        sys.exit(0)