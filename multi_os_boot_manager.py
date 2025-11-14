"""
Pandora AIOS Multi-OS Boot Manager (Hackintosh-Style)
------------------------------------------------------
Provides a sophisticated boot menu and OS switching system similar to Hackintosh/Clover.
Allows seamless switching between multiple operating systems while maintaining
Pandora AIOS functionality across all platforms.

Features:
- Multi-OS detection and management
- Boot menu with theme support
- OS-specific configuration profiles
- Seamless switching between Windows, Linux, macOS
- VM integration support
- Boot parameters and kernel options
- Boot history and preferences
- Safe boot modes per OS

Philosophy: Unity in diversity, seamless transitions, respect for each OS ecosystem
"""

import os
import sys
import json
import subprocess
import platform
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

class OSProfile:
    """Represents an operating system installation"""
    
    def __init__(self, name: str, os_type: str, boot_path: str, partition: str = ""):
        self.name = name
        self.os_type = os_type  # windows, linux, macos, bsd
        self.boot_path = boot_path
        self.partition = partition
        self.boot_params = []
        self.kernel_options = {}
        self.custom_config = {}
        self.last_booted = None
        self.boot_count = 0
        self.is_active = False
        self.uuid = hashlib.md5(f"{name}{boot_path}".encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "os_type": self.os_type,
            "boot_path": self.boot_path,
            "partition": self.partition,
            "boot_params": self.boot_params,
            "kernel_options": self.kernel_options,
            "custom_config": self.custom_config,
            "last_booted": self.last_booted,
            "boot_count": self.boot_count,
            "is_active": self.is_active,
            "uuid": self.uuid
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'OSProfile':
        """Create from dictionary"""
        profile = OSProfile(
            name=data["name"],
            os_type=data["os_type"],
            boot_path=data["boot_path"],
            partition=data.get("partition", "")
        )
        profile.boot_params = data.get("boot_params", [])
        profile.kernel_options = data.get("kernel_options", {})
        profile.custom_config = data.get("custom_config", {})
        profile.last_booted = data.get("last_booted")
        profile.boot_count = data.get("boot_count", 0)
        profile.is_active = data.get("is_active", False)
        profile.uuid = data.get("uuid", profile.uuid)
        return profile


class MultiOSBootManager:
    """Main boot manager for handling multiple operating systems"""
    
    def __init__(self, config_path: str = "/etc/pandora/boot_config.json"):
        self.config_path = config_path
        self.os_profiles: List[OSProfile] = []
        self.current_os = None
        self.default_os = None
        self.boot_timeout = 10  # seconds
        self.theme = "pandora_dark"
        self.last_boot_choice = None
        self.boot_history = []
        
        # Fallback to temp if /etc not writable
        if not os.path.exists(os.path.dirname(config_path)):
            self.config_path = os.path.expanduser("~/.pandora_boot_config.json")
        
        self._detect_current_os()
        self.load_config()
    
    def _detect_current_os(self):
        """Detect the currently running OS"""
        system = platform.system()
        release = platform.release()
        
        if system == "Windows":
            self.current_os = "Windows"
        elif system == "Linux":
            self.current_os = "Linux"
        elif system == "Darwin":
            self.current_os = "macOS"
        else:
            self.current_os = system
    
    def load_config(self):
        """Load boot configuration"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                
                self.os_profiles = [OSProfile.from_dict(p) for p in data.get("profiles", [])]
                self.default_os = data.get("default_os")
                self.boot_timeout = data.get("boot_timeout", 10)
                self.theme = data.get("theme", "pandora_dark")
                self.boot_history = data.get("boot_history", [])
                
                print(f"[INFO] Loaded {len(self.os_profiles)} OS profiles")
                
            except Exception as e:
                print(f"[ERROR] Failed to load config: {e}")
                self._create_default_config()
        else:
            self._create_default_config()
    
    def save_config(self):
        """Save boot configuration"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            data = {
                "profiles": [p.to_dict() for p in self.os_profiles],
                "default_os": self.default_os,
                "boot_timeout": self.boot_timeout,
                "theme": self.theme,
                "boot_history": self.boot_history[-100:],  # Keep last 100 boots
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"[INFO] Configuration saved to {self.config_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save config: {e}")
    
    def _create_default_config(self):
        """Create default configuration with detected OS"""
        print("[INFO] Creating default boot configuration...")
        
        # Add current OS as first profile
        if self.current_os:
            current_profile = OSProfile(
                name=f"{self.current_os} (Current)",
                os_type=self.current_os.lower(),
                boot_path="/boot",
                partition="auto"
            )
            current_profile.is_active = True
            self.os_profiles.append(current_profile)
            self.default_os = current_profile.uuid
        
        self.save_config()
    
    def detect_os_installations(self) -> List[OSProfile]:
        """Automatically detect OS installations on the system"""
        detected = []
        
        print("[INFO] Scanning for OS installations...")
        
        # Detect Linux installations
        detected.extend(self._detect_linux_installs())
        
        # Detect Windows installations
        detected.extend(self._detect_windows_installs())
        
        # Detect macOS installations (if on Mac or Hackintosh)
        detected.extend(self._detect_macos_installs())
        
        # Detect BSD installations
        detected.extend(self._detect_bsd_installs())
        
        print(f"[INFO] Detected {len(detected)} OS installations")
        
        return detected
    
    def _detect_linux_installs(self) -> List[OSProfile]:
        """Detect Linux installations"""
        linux_installs = []
        
        try:
            # Check /etc/fstab and mounted partitions
            if os.path.exists("/proc/mounts"):
                with open("/proc/mounts", "r") as f:
                    mounts = f.readlines()
                
                for mount in mounts:
                    parts = mount.split()
                    if len(parts) >= 2:
                        device, mountpoint = parts[0], parts[1]
                        
                        # Check for Linux root partitions
                        if mountpoint == "/":
                            # Try to detect distribution
                            distro_name = self._detect_linux_distro(mountpoint)
                            profile = OSProfile(
                                name=f"Linux - {distro_name}",
                                os_type="linux",
                                boot_path=mountpoint,
                                partition=device
                            )
                            linux_installs.append(profile)
            
            # Check for other Linux installations in /boot
            if os.path.exists("/boot/grub/grub.cfg"):
                # Parse GRUB config for other Linux installations
                pass  # Implement GRUB parsing if needed
                
        except Exception as e:
            print(f"[WARNING] Linux detection error: {e}")
        
        return linux_installs
    
    def _detect_linux_distro(self, root_path: str) -> str:
        """Detect Linux distribution name"""
        release_files = [
            "/etc/os-release",
            "/etc/lsb-release",
            "/etc/redhat-release",
            "/etc/debian_version"
        ]
        
        for rel_file in release_files:
            full_path = os.path.join(root_path, rel_file.lstrip('/'))
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                        if "NAME=" in content:
                            for line in content.split('\n'):
                                if line.startswith("NAME="):
                                    return line.split('=')[1].strip('"')
                        return "Linux"
                except:
                    pass
        
        return "Linux"
    
    def _detect_windows_installs(self) -> List[OSProfile]:
        """Detect Windows installations"""
        windows_installs = []
        
        try:
            # On Linux, check mounted NTFS partitions
            if platform.system() == "Linux":
                result = subprocess.run(
                    ["lsblk", "-J", "-o", "NAME,FSTYPE,LABEL,MOUNTPOINT"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    for device in data.get("blockdevices", []):
                        if device.get("fstype") == "ntfs":
                            # Likely Windows partition
                            label = device.get("label", "Windows")
                            profile = OSProfile(
                                name=f"Windows - {label}",
                                os_type="windows",
                                boot_path=device.get("mountpoint", ""),
                                partition=f"/dev/{device['name']}"
                            )
                            windows_installs.append(profile)
            
            # On Windows, current installation
            elif platform.system() == "Windows":
                profile = OSProfile(
                    name="Windows (Current)",
                    os_type="windows",
                    boot_path="C:\\",
                    partition="C:"
                )
                windows_installs.append(profile)
                
        except Exception as e:
            print(f"[WARNING] Windows detection error: {e}")
        
        return windows_installs
    
    def _detect_macos_installs(self) -> List[OSProfile]:
        """Detect macOS installations"""
        macos_installs = []
        
        try:
            # On macOS or Hackintosh
            if platform.system() == "Darwin":
                profile = OSProfile(
                    name="macOS (Current)",
                    os_type="macos",
                    boot_path="/",
                    partition="/"
                )
                macos_installs.append(profile)
            
            # Check for macOS on other partitions (Hackintosh scenario)
            # Look for HFS+ or APFS partitions
            if platform.system() == "Linux":
                result = subprocess.run(
                    ["lsblk", "-J", "-o", "NAME,FSTYPE,LABEL"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    for device in data.get("blockdevices", []):
                        fstype = device.get("fstype", "")
                        if fstype in ["hfsplus", "apfs"]:
                            label = device.get("label", "macOS")
                            profile = OSProfile(
                                name=f"macOS - {label}",
                                os_type="macos",
                                boot_path="",
                                partition=f"/dev/{device['name']}"
                            )
                            macos_installs.append(profile)
                            
        except Exception as e:
            print(f"[WARNING] macOS detection error: {e}")
        
        return macos_installs
    
    def _detect_bsd_installs(self) -> List[OSProfile]:
        """Detect BSD installations"""
        bsd_installs = []
        
        try:
            # Check for BSD partitions (UFS)
            if platform.system() == "Linux":
                result = subprocess.run(
                    ["lsblk", "-J", "-o", "NAME,FSTYPE,LABEL"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    for device in data.get("blockdevices", []):
                        if device.get("fstype") == "ufs":
                            label = device.get("label", "BSD")
                            profile = OSProfile(
                                name=f"BSD - {label}",
                                os_type="bsd",
                                boot_path="",
                                partition=f"/dev/{device['name']}"
                            )
                            bsd_installs.append(profile)
                            
        except Exception as e:
            print(f"[WARNING] BSD detection error: {e}")
        
        return bsd_installs
    
    def add_os_profile(self, profile: OSProfile):
        """Add a new OS profile"""
        # Check if already exists
        for existing in self.os_profiles:
            if existing.uuid == profile.uuid:
                print(f"[WARNING] Profile already exists: {profile.name}")
                return
        
        self.os_profiles.append(profile)
        print(f"[INFO] Added OS profile: {profile.name}")
        
        # Set as default if it's the first
        if len(self.os_profiles) == 1:
            self.default_os = profile.uuid
        
        self.save_config()
    
    def remove_os_profile(self, uuid: str):
        """Remove an OS profile"""
        self.os_profiles = [p for p in self.os_profiles if p.uuid != uuid]
        
        if self.default_os == uuid:
            self.default_os = self.os_profiles[0].uuid if self.os_profiles else None
        
        self.save_config()
    
    def set_default_os(self, uuid: str):
        """Set default OS"""
        for profile in self.os_profiles:
            if profile.uuid == uuid:
                self.default_os = uuid
                print(f"[INFO] Default OS set to: {profile.name}")
                self.save_config()
                return
        
        print(f"[ERROR] OS profile not found: {uuid}")
    
    def display_boot_menu(self) -> Optional[OSProfile]:
        """Display interactive boot menu"""
        print("\n" + "="*70)
        print(" " * 20 + "Pandora AIOS Boot Manager")
        print("="*70)
        print()
        
        if not self.os_profiles:
            print("No OS profiles configured!")
            return None
        
        # Display OS options
        for i, profile in enumerate(self.os_profiles, 1):
            default_marker = " [DEFAULT]" if profile.uuid == self.default_os else ""
            active_marker = " (CURRENT)" if profile.is_active else ""
            boot_info = f" - Booted {profile.boot_count} times"
            
            print(f"  {i}. {profile.name}{default_marker}{active_marker}")
            print(f"     Type: {profile.os_type.upper()} | Partition: {profile.partition}")
            print(f"     Last booted: {profile.last_booted or 'Never'}{boot_info}")
            print()
        
        print("="*70)
        print(f"Options:")
        print(f"  [1-{len(self.os_profiles)}] - Select OS")
        print(f"  [S] - System Settings")
        print(f"  [D] - Detect OS Installations")
        print(f"  [R] - Refresh/Reboot")
        print(f"  [Q] - Quit")
        print("="*70)
        
        # Get user choice
        try:
            choice = input(f"\nSelect option (timeout in {self.boot_timeout}s): ").strip().upper()
            
            if choice == 'Q':
                return None
            elif choice == 'S':
                self.settings_menu()
                return self.display_boot_menu()
            elif choice == 'D':
                self.detect_and_add_os()
                return self.display_boot_menu()
            elif choice == 'R':
                return self._get_default_profile()
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(self.os_profiles):
                    return self.os_profiles[idx]
            
            print("[ERROR] Invalid choice")
            return self.display_boot_menu()
            
        except KeyboardInterrupt:
            print("\n[INFO] Boot cancelled")
            return None
    
    def settings_menu(self):
        """Display settings menu"""
        print("\n" + "="*70)
        print(" " * 25 + "Boot Settings")
        print("="*70)
        print(f"1. Set default OS (current: {self._get_os_name_by_uuid(self.default_os)})")
        print(f"2. Set boot timeout (current: {self.boot_timeout}s)")
        print(f"3. Change theme (current: {self.theme})")
        print(f"4. View boot history")
        print(f"5. Add custom OS entry")
        print(f"6. Remove OS entry")
        print("0. Back to main menu")
        print("="*70)
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            self._set_default_os_interactive()
        elif choice == '2':
            self._set_timeout_interactive()
        elif choice == '3':
            self._set_theme_interactive()
        elif choice == '4':
            self._view_boot_history()
        elif choice == '5':
            self._add_custom_os_interactive()
        elif choice == '6':
            self._remove_os_interactive()
    
    def _get_os_name_by_uuid(self, uuid: str) -> str:
        """Get OS name by UUID"""
        for profile in self.os_profiles:
            if profile.uuid == uuid:
                return profile.name
        return "None"
    
    def _get_default_profile(self) -> Optional[OSProfile]:
        """Get default OS profile"""
        for profile in self.os_profiles:
            if profile.uuid == self.default_os:
                return profile
        return self.os_profiles[0] if self.os_profiles else None
    
    def _set_default_os_interactive(self):
        """Interactively set default OS"""
        print("\nAvailable OS:")
        for i, profile in enumerate(self.os_profiles, 1):
            print(f"  {i}. {profile.name}")
        
        try:
            choice = int(input("\nSelect default OS number: ")) - 1
            if 0 <= choice < len(self.os_profiles):
                self.set_default_os(self.os_profiles[choice].uuid)
        except:
            print("[ERROR] Invalid selection")
    
    def _set_timeout_interactive(self):
        """Interactively set boot timeout"""
        try:
            timeout = int(input(f"\nEnter timeout in seconds (current: {self.boot_timeout}): "))
            if timeout >= 0:
                self.boot_timeout = timeout
                self.save_config()
                print(f"[INFO] Boot timeout set to {timeout}s")
        except:
            print("[ERROR] Invalid timeout value")
    
    def _set_theme_interactive(self):
        """Interactively set theme"""
        themes = ["pandora_dark", "pandora_light", "clover", "grub", "minimal"]
        print("\nAvailable themes:")
        for i, theme in enumerate(themes, 1):
            print(f"  {i}. {theme}")
        
        try:
            choice = int(input("\nSelect theme number: ")) - 1
            if 0 <= choice < len(themes):
                self.theme = themes[choice]
                self.save_config()
                print(f"[INFO] Theme set to {self.theme}")
        except:
            print("[ERROR] Invalid selection")
    
    def _view_boot_history(self):
        """View boot history"""
        print("\n" + "="*70)
        print(" " * 25 + "Boot History")
        print("="*70)
        
        if not self.boot_history:
            print("No boot history available")
        else:
            for i, entry in enumerate(reversed(self.boot_history[-20:]), 1):
                print(f"{i}. {entry['timestamp']} - {entry['os_name']} ({entry['os_type']})")
        
        input("\nPress Enter to continue...")
    
    def _add_custom_os_interactive(self):
        """Add custom OS entry interactively"""
        print("\n=== Add Custom OS Entry ===")
        name = input("OS Name: ").strip()
        os_type = input("OS Type (windows/linux/macos/bsd): ").strip().lower()
        boot_path = input("Boot Path: ").strip()
        partition = input("Partition (optional): ").strip()
        
        if name and os_type and boot_path:
            profile = OSProfile(name, os_type, boot_path, partition)
            self.add_os_profile(profile)
        else:
            print("[ERROR] Invalid input")
    
    def _remove_os_interactive(self):
        """Remove OS entry interactively"""
        print("\n=== Remove OS Entry ===")
        for i, profile in enumerate(self.os_profiles, 1):
            print(f"  {i}. {profile.name}")
        
        try:
            choice = int(input("\nSelect OS to remove: ")) - 1
            if 0 <= choice < len(self.os_profiles):
                profile = self.os_profiles[choice]
                confirm = input(f"Remove '{profile.name}'? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    self.remove_os_profile(profile.uuid)
                    print(f"[INFO] Removed {profile.name}")
        except:
            print("[ERROR] Invalid selection")
    
    def detect_and_add_os(self):
        """Detect and add new OS installations"""
        print("\n[INFO] Detecting OS installations...")
        detected = self.detect_os_installations()
        
        new_count = 0
        for profile in detected:
            # Check if already exists
            exists = any(p.uuid == profile.uuid for p in self.os_profiles)
            if not exists:
                self.add_os_profile(profile)
                new_count += 1
        
        print(f"[INFO] Added {new_count} new OS profiles")
        input("\nPress Enter to continue...")
    
    def boot_os(self, profile: OSProfile):
        """Boot selected OS"""
        print(f"\n[INFO] Preparing to boot: {profile.name}")
        
        # Update boot statistics
        profile.last_booted = datetime.now().isoformat()
        profile.boot_count += 1
        
        # Add to boot history
        self.boot_history.append({
            "timestamp": datetime.now().isoformat(),
            "os_name": profile.name,
            "os_type": profile.os_type,
            "uuid": profile.uuid
        })
        
        self.save_config()
        
        # Execute boot command based on OS type
        if profile.os_type == "windows":
            self._boot_windows(profile)
        elif profile.os_type == "linux":
            self._boot_linux(profile)
        elif profile.os_type == "macos":
            self._boot_macos(profile)
        elif profile.os_type == "bsd":
            self._boot_bsd(profile)
        else:
            print(f"[ERROR] Unsupported OS type: {profile.os_type}")
    
    def _boot_windows(self, profile: OSProfile):
        """Boot Windows OS"""
        print(f"[INFO] Booting Windows from {profile.partition}")
        
        # On Linux, use GRUB or bootctl
        if platform.system() == "Linux":
            # Try to use efibootmgr or grub-reboot
            try:
                subprocess.run(["systemctl", "reboot", "--boot-loader-entry=windows"])
            except:
                print("[ERROR] Unable to boot Windows. Manual reboot required.")
        else:
            print("[INFO] Already on Windows or boot not supported from current OS")
    
    def _boot_linux(self, profile: OSProfile):
        """Boot Linux OS"""
        print(f"[INFO] Booting Linux from {profile.boot_path}")
        
        # If already on Linux, might need to switch kernel or reboot
        if platform.system() == "Linux":
            print("[INFO] Already on Linux. Use GRUB to select different installation.")
        else:
            # From other OS, try to reboot to Linux
            try:
                subprocess.run(["systemctl", "reboot"])
            except:
                print("[ERROR] Unable to boot Linux. Manual reboot required.")
    
    def _boot_macos(self, profile: OSProfile):
        """Boot macOS"""
        print(f"[INFO] Booting macOS from {profile.partition}")
        
        # Use bless command on macOS or boot from UEFI
        if platform.system() == "Darwin":
            print("[INFO] Already on macOS")
        else:
            print("[INFO] Reboot and select macOS from boot menu")
    
    def _boot_bsd(self, profile: OSProfile):
        """Boot BSD OS"""
        print(f"[INFO] Booting BSD from {profile.partition}")
        print("[INFO] Manual boot selection required")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pandora AIOS Multi-OS Boot Manager")
    parser.add_argument("--detect", action="store_true", help="Auto-detect OS installations")
    parser.add_argument("--list", action="store_true", help="List configured OS profiles")
    parser.add_argument("--boot", type=str, help="Boot specific OS by name or UUID")
    parser.add_argument("--default", type=str, help="Set default OS by UUID")
    parser.add_argument("--config", type=str, help="Config file path")
    
    args = parser.parse_args()
    
    config_path = args.config or "/etc/pandora/boot_config.json"
    manager = MultiOSBootManager(config_path=config_path)
    
    if args.detect:
        print("\n=== Detecting OS Installations ===")
        detected = manager.detect_os_installations()
        for profile in detected:
            print(f"  - {profile.name} ({profile.os_type}) on {profile.partition}")
            manager.add_os_profile(profile)
    
    elif args.list:
        print("\n=== Configured OS Profiles ===")
        for profile in manager.os_profiles:
            default = " [DEFAULT]" if profile.uuid == manager.default_os else ""
            print(f"  - {profile.name}{default}")
            print(f"    Type: {profile.os_type} | UUID: {profile.uuid}")
            print(f"    Partition: {profile.partition}")
            print(f"    Boot count: {profile.boot_count}")
            print()
    
    elif args.default:
        manager.set_default_os(args.default)
    
    elif args.boot:
        # Find OS by name or UUID
        target_profile = None
        for profile in manager.os_profiles:
            if profile.uuid == args.boot or profile.name == args.boot:
                target_profile = profile
                break
        
        if target_profile:
            manager.boot_os(target_profile)
        else:
            print(f"[ERROR] OS not found: {args.boot}")
    
    else:
        # Interactive mode
        selected = manager.display_boot_menu()
        if selected:
            manager.boot_os(selected)


if __name__ == "__main__":
    main()
