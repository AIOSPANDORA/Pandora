# Pandora Launcher - Quick Start Guide

## What is PandoraLauncher?

PandoraLauncher is a universal executable that provides easy access to all Pandora Quantum System features without needing to know Python or command-line operations.

## Quick Start (3 Steps)

### 1. Download or Build

**Option A: Use Pre-built Executable** (Recommended)
- Download `PandoraLauncher.exe` (Windows), `PandoraLauncher` (Linux), or `PandoraLauncher.app` (macOS)
- No installation required - just run it!

**Option B: Build from Source**
```bash
# Install dependencies first
pip install pyinstaller numpy scipy

# Then build
./build_executable.sh      # Linux/macOS
build_executable.bat        # Windows
```

### 2. Run

**Windows**
- Double-click `PandoraLauncher.exe`
- Or from Command Prompt: `PandoraLauncher.exe`

**Linux/macOS**
```bash
chmod +x PandoraLauncher
./PandoraLauncher
```

### 3. Explore

Once launched, you'll see a menu with options:

```
1. Run Tests          → Verify everything works
2. Demo               → See Hamiltonian profiles in action
3. Security Example   → Quantum encryption demo
4. Interactive Mode   → Experiment with quantum operations
5. Profile Selector   → Try different quantum profiles
6. System Info        → Check your setup
```

## First-Time Users

**Start here:**
1. Launch PandoraLauncher
2. Select option **1** to run tests
3. Select option **2** to see the demo
4. Select option **5** to explore different quantum profiles

## What Each Menu Option Does

### 1. Run Quantum Profile Tests
- Runs 40+ automated tests
- Verifies Hamiltonian operations
- Tests all 5 quantum profiles
- Takes ~10 seconds

**When to use**: First time setup, after updates, to verify installation

### 2. Launch Quantum Hamiltonian Demo
- Shows all 5 quantum profiles
- Demonstrates basic operations
- ML optimization example
- Takes ~30 seconds

**When to use**: To understand what the system can do

### 3. Run Security Integration Example
- Quantum random number generation
- Encryption key derivation
- Authentication token creation
- Takes ~15 seconds

**When to use**: To see real-world applications

### 4. Interactive Quantum Processor Session
- Opens Python console
- Pre-loaded quantum processor
- Experiment with commands
- Type `exit()` to return

**When to use**: For advanced users who want to experiment

### 5. Select and Launch Quantum Profile
- Choose from 5 specialized profiles:
  - **Alternative**: For exploration via superposition
  - **Castle**: For stable, defensive operations
  - **Hive**: For collective, distributed behavior
  - **Empire**: For hierarchical control
  - **Omega**: For balanced, general-purpose use

**When to use**: To understand profile differences

### 6. System Information
- Shows platform details
- Lists installed modules
- Displays available profiles

**When to use**: To troubleshoot or check configuration

## The 5 Quantum Profiles

| Profile | Best For | Energy | Characteristics |
|---------|----------|--------|-----------------|
| **Alternative** | Quantum search, exploration | Low | High superposition, X/Y operators |
| **Castle** | Stable storage, protection | High | Strong Z barriers, defensive |
| **Hive** | Distributed systems | Low | All-to-all coupling, collective |
| **Empire** | Centralized control | Medium | Hierarchical, cascading effects |
| **Omega** | General computation | Medium | Balanced, all operators |

## Example: Running Your First Demo

1. Launch PandoraLauncher
2. You'll see hardware detection and dependency checking
3. Select **2** (Launch Quantum Hamiltonian Demo)
4. Watch the demo run - it will show:
   - Basic Hamiltonian operations
   - All 5 profiles compared
   - ML optimization in action
   - Profile management
5. Press Enter when done to return to menu

## Common Questions

**Q: Do I need Python installed?**
A: No! The executable includes everything needed.

**Q: Can I use this on any computer?**
A: Yes, as long as the architecture matches (most are 64-bit).

**Q: How big is the executable?**
A: 50-100 MB (includes Python runtime and all dependencies).

**Q: Is it safe?**
A: Yes. Built from open source code, 0 security vulnerabilities detected.

**Q: Can I distribute this?**
A: Yes. The executable can be copied to any compatible system.

**Q: What if I get an error?**
A: Run option **1** (tests) to diagnose, or check option **6** (system info).

## Pinning to Top of Folder

To keep PandoraLauncher at the top of your folder:

**Windows**
1. Rename to `!PandoraLauncher.exe` (the `!` makes it sort first)
2. Or pin to taskbar: Right-click → Pin to Taskbar

**Linux**
1. Use file manager's "Keep at top" feature
2. Or rename to `0-PandoraLauncher`

**macOS**
1. Drag to Applications folder
2. Or use Command+J to adjust sort settings

## Making it Executable (Linux/macOS)

If the file isn't executable:
```bash
chmod +x PandoraLauncher
```

## Troubleshooting

**"Permission denied" error**
```bash
chmod +x PandoraLauncher    # Linux/macOS
```

**"Missing dependencies" message**
- This appears if running as Python script (not executable)
- Install: `pip install numpy scipy`
- Or build executable which bundles everything

**Menu doesn't appear**
- Check console output for errors
- Run tests (option 1) to diagnose
- See LAUNCHER_README.md for details

**Executable won't run**
- Windows: Install Visual C++ Redistributable
- Linux: Check glibc version (needs 2.17+)
- macOS: Check macOS version (needs 10.13+)

## Next Steps

After exploring the launcher:
1. Check `quantum_profiles/README.md` for API documentation
2. Look at test files to see code examples
3. Explore the demo scripts
4. Try the interactive mode (option 4)

## Getting Help

1. Run option **6** to see system information
2. Run option **1** to verify installation
3. Check `LAUNCHER_README.md` for detailed documentation
4. Review error messages shown in the launcher

## Advanced: Building from Source

See `LAUNCHER_README.md` for:
- Custom icon configuration
- Reducing file size
- Building for different architectures
- Creating desktop shortcuts
- Distribution guidelines

---

**Ready to start?** Just double-click PandoraLauncher and select option 2 for a demo!
