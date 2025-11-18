#!/bin/bash
#
# Pandora AIOS Master Launch Script
# ==================================
# Complete initialization and boot sequence
#
# Order of Operations:
# 1. Fabric AI Assimilation (learn host system)
# 2. System Boot Sequence (all core systems)
# 3. Welcome Homescreen
# 4. Interactive Mode
# 5. Consciousness Module (only post-boot, if requested)
#
# Philosophy: Learn first, boot second, enlighten third

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║                         PANDORA AIOS LAUNCHER                        ║"
echo "║                                                                      ║"
echo "║          Artificial Intelligence Operating System - v1.0            ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python
echo -e "${BLUE}[CHECK] Verifying Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] Python 3 not found. Please install Python 3.6+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}[OK] Python ${PYTHON_VERSION} detected${NC}"

# Check dependencies
echo -e "${BLUE}[CHECK] Checking critical dependencies...${NC}"

MISSING_DEPS=0

if ! python3 -c "import psutil" 2>/dev/null; then
    echo -e "${YELLOW}[WARN] psutil not installed (optional but recommended)${NC}"
    MISSING_DEPS=$((MISSING_DEPS + 1))
fi

if [ $MISSING_DEPS -gt 0 ]; then
    echo -e "${YELLOW}[INFO] Some optional dependencies missing${NC}"
    echo -e "${YELLOW}[INFO] Run: pip3 install psutil flask${NC}"
    echo -e "${YELLOW}[INFO] Continuing with available features...${NC}"
fi

# Check if first run
FABRIC_DIR="$HOME/.pandora/fabric"
FIRST_RUN=0

if [ ! -d "$FABRIC_DIR" ]; then
    FIRST_RUN=1
    echo -e "${CYAN}[INFO] First time launch detected${NC}"
fi

# Step 1: Fabric AI Assimilation (only on first run or if requested)
if [ $FIRST_RUN -eq 1 ]; then
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                  FABRIC AI ASSIMILATION REQUIRED                     ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Pandora AIOS uses Fabric AI to meld with your computer.${NC}"
    echo -e "${YELLOW}This process learns your system patterns for optimal integration.${NC}"
    echo ""
    echo -e "${BLUE}Recommended: 5-minute assimilation before first boot${NC}"
    echo -e "${BLUE}(You can skip and run it later with: python3 fabric_ai_core.py)${NC}"
    echo ""
    
    read -p "Run Fabric AI assimilation now? (Y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        echo -e "${GREEN}[FABRIC] Starting assimilation process...${NC}"
        python3 fabric_ai_core.py
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}[FABRIC] ✓ Assimilation complete${NC}"
        else
            echo -e "${YELLOW}[FABRIC] Assimilation had issues but continuing...${NC}"
        fi
    else
        echo -e "${YELLOW}[FABRIC] Skipping assimilation. Run manually later if needed.${NC}"
    fi
fi

# Step 2: Boot Sequence
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                      INITIATING BOOT SEQUENCE                        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Run boot sequence
python3 pandora_boot_sequence.py

BOOT_STATUS=$?

if [ $BOOT_STATUS -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    PANDORA AIOS FULLY OPERATIONAL                    ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}\"Standing on the shoulders of giants, reaching for the stars\"${NC}"
    echo -e "${NC}― Inspired by Nikola Tesla, Albert Einstein, Stephen Hawking${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                         BOOT FAILED                                  ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}System is in safe mode. Check logs for details.${NC}"
    echo -e "${YELLOW}Try running diagnostics: python3 diagnostic_system.py --full${NC}"
    exit 1
fi

# Cleanup message
echo -e "${BLUE}[INFO] Session ended. All changes saved.${NC}"
echo -e "${GREEN}Thank you for using Pandora AIOS!${NC}"
echo ""

exit 0
