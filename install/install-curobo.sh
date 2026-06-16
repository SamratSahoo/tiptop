#!/bin/bash

set -eo pipefail

# Verify we're running from project root (check for tiptop-specific files)
if [ ! -f "tiptop/__init__.py" ]; then
    echo "ERROR: This script must be run from the tiptop project root directory"
    echo "Please run: pixi run install-curobo"
    exit 1
fi

# cuRobo is now a sibling submodule of the parent tamp-vla checkout
# (tamp-vla/curobo), not vendored under tiptop/. Its clone/checkout is managed by
# the parent repo's git submodule, so this script only builds + installs it.
# Override CUROBO_DIR if your layout differs.
CUROBO_DIR="${CUROBO_DIR:-../curobo}"

echo "==> Installing curobo from $CUROBO_DIR"

if [ ! -d "$CUROBO_DIR/src/curobo" ]; then
    echo "ERROR: cuRobo submodule not found at $CUROBO_DIR"
    echo "Initialize it from the tamp-vla root with:"
    echo "    git submodule update --init curobo"
    exit 1
fi

echo "✓ curobo at: $(cd "$CUROBO_DIR" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# pip install (builds CUDA kernels too)
echo "Installing curobo (might take 5-20 minutes)..."
pip install -e "$CUROBO_DIR" --no-build-isolation --no-deps

echo "✓ curobo installed successfully"
