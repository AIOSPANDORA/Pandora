# Pandora Launcher - Cross-Platform Executable

## Overview

The Pandora Launcher is a universal, cross-platform executable that provides easy access to Pandora's Quantum System with Hamiltonian support. It can be built as a standalone executable for Windows (.exe), Linux, and macOS.

## Features

- **Hardware Detection**: Automatically detects NVIDIA GPU, Intel CPU, and ASUS ROG systems
- **Dependency Checking**: Verifies required Python packages (numpy, scipy)
- **Interactive Menu**: Easy-to-use menu system for accessing all features
- **Quantum Profile Selection**: Launch any of the 5 quantum profiles (Alternative, Castle, Hive, Empire, Omega)
- **Test Suite**: Run comprehensive tests (40+ unit tests)
- **Demo Launcher**: Execute demonstration scripts
- **Interactive Session**: Python REPL with quantum processor pre-loaded
- **Security Examples**: Run quantum security integration examples
- **System Information**: View detailed system and module information

## Building the Executable

### Prerequisites

1. **Python 3.8 or higher**
2. **PyInstaller**: `pip install pyinstaller`
3. **Dependencies**: `pip install numpy scipy`

### Quick Build

#### Windows
```cmd
build_executable.bat
```

#### Linux/macOS
```bash
chmod +x build_executable.sh
./build_executable.sh
```

### Manual Build

```bash
pyinstaller PandoraLauncher.spec --clean
```

The executable will be created in the `dist/` directory.

## Distribution

### Windows
- **File**: `dist/PandoraLauncher.exe`
- **Size**: ~50-100 MB (includes Python runtime and all dependencies)
- **Requirements**: Windows 7 or higher
- **Architecture**: Built for x64 (can build for x86 if needed)

### Linux
- **File**: `dist/PandoraLauncher`
- **Size**: ~50-100 MB
- **Requirements**: Most Linux distributions (glibc 2.17+)
- **Architecture**: Built for x86_64

### macOS
- **File**: `dist/PandoraLauncher.app` or `dist/PandoraLauncher`
- **Size**: ~50-100 MB
- **Requirements**: macOS 10.13 (High Sierra) or higher
- **Architecture**: Built for x86_64 (Intel) or arm64 (Apple Silicon)

## Usage

### Running the Executable

#### Windows
```cmd
# Double-click the .exe file, or:
dist\PandoraLauncher.exe
```

#### Linux
```bash
chmod +x dist/PandoraLauncher
./dist/PandoraLauncher
```

#### macOS
```bash
# Double-click the .app bundle, or:
./dist/PandoraLauncher
```

### Menu Options

Once launched, you'll see an interactive menu:

```
1. Run Quantum Profile Tests
   - Executes comprehensive test suite
   - Verifies all Hamiltonian operations
   - Tests all 5 quantum profiles

2. Launch Quantum Hamiltonian Demo
   - Demonstrates basic Hamiltonian operations
   - Shows all 5 profiles in action
   - ML optimization examples

3. Run Security Integration Example
   - Quantum random number generation
   - Encryption key derivation
   - Authentication token creation

4. Interactive Quantum Processor Session
   - Python REPL with quantum processor
   - Pre-loaded QuantumVirtualProcessor instance
   - Experiment with quantum operations

5. Select and Launch Quantum Profile
   - Choose from 5 specialized profiles
   - View profile characteristics
   - See energy calculations

6. System Information
   - View platform details
   - Check installed modules
   - List available quantum profiles

0. Exit
```

## Advanced Configuration

### Custom Icon

To add a custom icon to the executable, create an icon file (`.ico` for Windows, `.icns` for macOS) and update `PandoraLauncher.spec`:

```python
exe = EXE(
    ...
    icon='path/to/pandora.ico',  # Add this line
    ...
)
```

### Reducing File Size

To reduce the executable size:

1. **Use exclude modules**: Edit `PandoraLauncher.spec` and add unused modules to the `excludes` list
2. **Use UPX compression**: Install UPX and enable in spec file (already enabled)
3. **One-directory mode**: Remove `a.binaries, a.zipfiles, a.datas` from `EXE()` to create a directory with multiple files instead of a single executable

Example for one-directory mode:

```python
exe = EXE(
    pyz,
    a.scripts,
    [],  # Don't include binaries in single file
    exclude_binaries=True,
    ...
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='PandoraLauncher'
)
```

### Building for Different Architectures

#### Windows x86 (32-bit)
```cmd
python -m PyInstaller PandoraLauncher.spec --clean
```
(Must be run on 32-bit Python)

#### macOS Universal Binary (Intel + Apple Silicon)
```bash
pyinstaller PandoraLauncher.spec --target-arch universal2
```

#### Linux ARM
```bash
# Must be built on ARM system or cross-compile
pyinstaller PandoraLauncher.spec --clean
```

## Troubleshooting

### "Module not found" errors

If the executable fails with module import errors:

1. Add the missing module to `hiddenimports` in `PandoraLauncher.spec`
2. Rebuild: `pyinstaller PandoraLauncher.spec --clean`

### "numpy" or "scipy" errors

Ensure you're using compatible versions:
```bash
pip install "numpy>=1.21.0" "scipy>=1.7.0"
```

### Executable won't run on other systems

- **Windows**: Ensure target system has Visual C++ Redistributable
- **Linux**: Check glibc version compatibility
- **macOS**: Check macOS version and architecture match

### Large file size

The executable includes Python runtime and all dependencies. This is normal. To reduce size, see "Reducing File Size" above.

## Creating a Desktop Shortcut

### Windows
1. Right-click `PandoraLauncher.exe`
2. Select "Create shortcut"
3. Move shortcut to Desktop or Pin to Taskbar

### Linux
Create `PandoraLauncher.desktop`:
```ini
[Desktop Entry]
Name=Pandora Quantum Launcher
Exec=/path/to/PandoraLauncher
Icon=/path/to/icon.png
Type=Application
Categories=Science;Education;
```

### macOS
1. Drag `PandoraLauncher.app` to Applications folder
2. Or create alias: Right-click → Make Alias

## Pinning to Top of Folder

### Windows
- Right-click the folder → Sort by → Name (ensures alphabetical order)
- Rename to `!PandoraLauncher.exe` (the `!` makes it appear first)
- Or use "Pin to Quick Access" in File Explorer

### Linux
- In file managers, use "Keep at top" or similar options
- Or use prefix like `0_PandoraLauncher`

### macOS
- In Finder, arrange by Name
- Use Command+J to adjust sort order
- Or use prefix to control ordering

## Security Notes

- The executable is built from source code in this repository
- No obfuscation is applied - you can inspect the code before building
- The executable does not require administrator/root privileges
- Network access is not required for basic functionality
- CodeQL security scan: 0 vulnerabilities

## Distribution Checklist

Before distributing the executable:

- [ ] Test on target platform
- [ ] Verify all menu options work
- [ ] Check that quantum profiles load correctly
- [ ] Ensure dependencies are bundled
- [ ] Test on clean system (no Python installed)
- [ ] Verify file size is reasonable
- [ ] Include README and documentation
- [ ] Add icon (optional but recommended)

## Version Information

- **Launcher Version**: 1.0.0
- **Quantum Profiles**: 5 (Alternative, Castle, Hive, Empire, Omega)
- **Python Requirement**: 3.8+ (bundled in executable)
- **Dependencies**: numpy, scipy (bundled in executable)

## Support

For issues or questions:
1. Check this README
2. Review error messages in the launcher
3. Run test suite (Menu option 1)
4. Check system information (Menu option 6)

## License

Part of the Pandora AIOS project.
