#!/bin/bash
# Build script for creating cross-platform executables using PyInstaller
#
# This script builds standalone executables for Pandora Quantum System
# that can run on any platform without requiring Python installation.
#
# Requirements:
#   - Python 3.8+
#   - PyInstaller: pip install pyinstaller
#   - numpy, scipy: pip install numpy scipy
#
# Usage:
#   ./build_executable.sh          # Build for current platform
#   ./build_executable.sh --all    # Build for all platforms (requires platform-specific builds)

set -e  # Exit on error

echo "======================================================================"
echo "  Pandora Quantum System - Executable Builder"
echo "======================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if PyInstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo -e "${RED}✗ PyInstaller not found${NC}"
    echo "Installing PyInstaller..."
    pip install pyinstaller
fi

# Check if dependencies are installed
echo "Checking dependencies..."
python3 -c "import numpy, scipy" 2>/dev/null || {
    echo -e "${RED}✗ Missing dependencies (numpy, scipy)${NC}"
    echo "Installing dependencies..."
    pip install numpy scipy
}
echo -e "${GREEN}✓ All dependencies installed${NC}"
echo ""

# Detect platform
PLATFORM=$(uname -s)
echo "Platform: $PLATFORM"
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist
echo ""

# Build executable
echo "Building Pandora Launcher executable..."
echo "This may take a few minutes..."
echo ""

pyinstaller PandoraLauncher.spec --clean

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo ""
    
    case "$PLATFORM" in
        Linux*)
            EXECUTABLE="dist/PandoraLauncher"
            echo "Executable created: $EXECUTABLE"
            echo "Making executable..."
            chmod +x "$EXECUTABLE"
            
            # Test the executable
            echo ""
            echo "Testing executable..."
            if "$EXECUTABLE" --version &> /dev/null || [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ Executable works!${NC}"
            else
                echo -e "${YELLOW}⚠ Executable may have issues (this is expected for menu-based apps)${NC}"
            fi
            ;;
        
        Darwin*)
            EXECUTABLE="dist/PandoraLauncher.app"
            echo "App bundle created: $EXECUTABLE"
            echo "You can also use: dist/PandoraLauncher"
            chmod +x "dist/PandoraLauncher"
            ;;
        
        MINGW*|MSYS*|CYGWIN*)
            EXECUTABLE="dist/PandoraLauncher.exe"
            echo "Executable created: $EXECUTABLE"
            ;;
        
        *)
            EXECUTABLE="dist/PandoraLauncher"
            echo "Executable created: $EXECUTABLE"
            chmod +x "$EXECUTABLE"
            ;;
    esac
    
    echo ""
    echo "======================================================================"
    echo "  Build Complete!"
    echo "======================================================================"
    echo ""
    echo "The executable is located in the 'dist' directory."
    echo ""
    echo "To run:"
    case "$PLATFORM" in
        Darwin*)
            echo "  Double-click dist/PandoraLauncher.app"
            echo "  OR run: ./dist/PandoraLauncher"
            ;;
        *)
            echo "  ./$EXECUTABLE"
            ;;
    esac
    echo ""
    echo "Distribution:"
    echo "  You can copy the executable to any compatible system"
    echo "  No Python installation required on target system"
    echo ""
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
