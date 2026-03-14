#!/bin/bash
# Build script for Ouroboros Sync Rust module

set -e  # Exit on error

echo "======================================"
echo "Building Ouroboros Sync Rust Module"
echo "======================================"

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "ERROR: Rust is not installed."
    echo "Please install Rust from https://rustup.rs/"
    exit 1
fi

# Check if we're in the correct directory
if [ ! -d "rust_modules/ouroboros_sync" ]; then
    echo "ERROR: rust_modules/ouroboros_sync directory not found."
    echo "Please run this script from the Pandora root directory."
    exit 1
fi

# Navigate to Rust module directory
echo ""
echo "Step 1: Building Rust module..."
cd rust_modules/ouroboros_sync

# Build the module
cargo build --release

if [ $? -ne 0 ]; then
    echo "ERROR: Cargo build failed."
    exit 1
fi

echo "✓ Rust module built successfully"

# Go back to root
cd ../..

# Copy the library
echo ""
echo "Step 2: Copying library to Python module directory..."

if [ -f "rust_modules/ouroboros_sync/target/release/libouroboros_sync.so" ]; then
    cp rust_modules/ouroboros_sync/target/release/libouroboros_sync.so ouroboros_sync.so
    echo "✓ Library copied successfully (Linux)"
elif [ -f "rust_modules/ouroboros_sync/target/release/libouroboros_sync.dylib" ]; then
    cp rust_modules/ouroboros_sync/target/release/libouroboros_sync.dylib ouroboros_sync.so
    echo "✓ Library copied successfully (macOS)"
elif [ -f "rust_modules/ouroboros_sync/target/release/ouroboros_sync.dll" ]; then
    cp rust_modules/ouroboros_sync/target/release/ouroboros_sync.dll ouroboros_sync.pyd
    echo "✓ Library copied successfully (Windows)"
else
    echo "ERROR: Could not find compiled library"
    exit 1
fi

# Test the module
echo ""
echo "Step 3: Testing the module..."
python3 -c "import ouroboros_sync; print('✓ Module imports successfully')" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Build completed successfully!"
    echo "======================================"
    echo ""
    echo "You can now use the module:"
    echo "  python3 ouroboros_sync_wrapper.py    # Run demo"
    echo "  python3 test_ouroboros_sync.py       # Run tests"
    echo "  python3 ouroboros_integration.py     # Run integration demo"
else
    echo ""
    echo "ERROR: Module import failed. Check Python version and dependencies."
    exit 1
fi
