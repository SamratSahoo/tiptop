#!/bin/bash

set -eo pipefail

# Verify we're running from project root (check for tiptop-specific files)
if [ ! -f "tiptop/__init__.py" ]; then
    echo "ERROR: This script must be run from the tiptop project root directory"
    echo "Please run: pixi run install-cutamp"
    exit 1
fi

# cuTAMP is now a sibling submodule of the parent tamp-vla checkout
# (tamp-vla/cuTAMP), not a separate clone vendored under tiptop/. Its
# clone/checkout is managed by the parent repo's git submodule, so this script
# only builds + installs it (mirrors install-curobo.sh). Override CUTAMP_DIR if
# your layout differs.
CUTAMP_DIR="${CUTAMP_DIR:-../cuTAMP}"

echo "==> Installing cuTAMP from $CUTAMP_DIR"

if [ ! -f "$CUTAMP_DIR/cutamp/__init__.py" ]; then
    echo "ERROR: cuTAMP submodule not found at $CUTAMP_DIR"
    echo "Initialize it from the tamp-vla root with:"
    echo "    git submodule update --init cuTAMP"
    exit 1
fi

echo "✓ cuTAMP at: $(cd "$CUTAMP_DIR" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# Sanity-check the submodule version against tiptop's pin (REQUIRED_CUTAMP_VERSION).
# tiptop's check_cutamp_version() enforces this at runtime; warn early if it won't match.
REQUIRED_VERSION=$(python -c "
import re
m = re.search(r'REQUIRED_CUTAMP_VERSION = \"([^\"]+)\"', open('tiptop/utils.py').read())
print(m.group(1) if m else '')
")
FOUND_VERSION=$(python -c "
import re
m = re.search(r'^version\s*=\s*\"([^\"]+)\"', open('$CUTAMP_DIR/pyproject.toml').read(), re.M)
print(m.group(1) if m else '')
")
if [ -n "$REQUIRED_VERSION" ] && [ "$FOUND_VERSION" != "$REQUIRED_VERSION" ]; then
    echo "WARNING: cuTAMP submodule version ('$FOUND_VERSION') != REQUIRED_CUTAMP_VERSION ('$REQUIRED_VERSION')."
    echo "         tiptop's check_cutamp_version() will fail at runtime. Update the cuTAMP submodule"
    echo "         (git -C $CUTAMP_DIR ...) or REQUIRED_CUTAMP_VERSION in tiptop/utils.py."
fi

# Editable install; --no-build-isolation + --no-deps so it uses the pixi env's
# torch/curobo and doesn't pull cuTAMP's own dependency set (provided by pixi.toml).
echo "Installing cuTAMP (editable)..."
pip install -e "$CUTAMP_DIR" --no-build-isolation --no-deps

echo "✓ cuTAMP installed successfully from $CUTAMP_DIR"
