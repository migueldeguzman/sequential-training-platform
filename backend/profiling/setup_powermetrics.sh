#!/bin/bash

# Energy Profiler - Setup powermetrics access
# This script configures passwordless sudo access for powermetrics

set -e

echo "==================================="
echo "Energy Profiler - powermetrics Setup"
echo "==================================="
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "ERROR: This script is only for macOS (Apple Silicon)"
    exit 1
fi

# Check if powermetrics exists
if ! command -v powermetrics &> /dev/null; then
    echo "ERROR: powermetrics command not found"
    echo "powermetrics should be available on macOS by default"
    exit 1
fi

# Get current user
CURRENT_USER=$(whoami)
echo "Current user: $CURRENT_USER"
echo ""

# Define the sudoers rule
SUDOERS_RULE="$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/bin/powermetrics"
SUDOERS_FILE="/etc/sudoers.d/powermetrics-$CURRENT_USER"

echo "This script will add the following rule to $SUDOERS_FILE:"
echo "  $SUDOERS_RULE"
echo ""
echo "This allows passwordless sudo access to powermetrics for the Energy Profiler."
echo ""

# Confirm with user
read -p "Do you want to proceed? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 0
fi

# Create sudoers entry
echo ""
echo "Creating sudoers entry (requires your password)..."
echo "$SUDOERS_RULE" | sudo tee "$SUDOERS_FILE" > /dev/null

# Set proper permissions
sudo chmod 0440 "$SUDOERS_FILE"

# Validate sudoers syntax
if sudo visudo -c -f "$SUDOERS_FILE" &> /dev/null; then
    echo ""
    echo "✓ Successfully configured passwordless sudo access for powermetrics"
    echo ""
    echo "Testing access..."
    if sudo -n powermetrics --help &> /dev/null; then
        echo "✓ Test successful - powermetrics is accessible without password"
        echo ""
        echo "Setup complete! The Energy Profiler can now use powermetrics."
    else
        echo "⚠ Test failed - please logout and login again for changes to take effect"
    fi
else
    echo "ERROR: Invalid sudoers syntax"
    sudo rm -f "$SUDOERS_FILE"
    exit 1
fi

echo ""
echo "==================================="
