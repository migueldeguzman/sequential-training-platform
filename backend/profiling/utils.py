"""
Energy Profiler - Utility Functions

Provides helper functions for the Energy Profiler system including
powermetrics availability checking and other utilities.
"""

import subprocess
import sys
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def check_powermetrics_access() -> Tuple[bool, str]:
    """
    Check if powermetrics is available with passwordless sudo access.

    Returns:
        Tuple[bool, str]: (is_available, message)
            - is_available: True if powermetrics can be run with sudo -n
            - message: Human-readable status message
    """
    # Check if running on macOS
    if sys.platform != 'darwin':
        return False, "powermetrics is only available on macOS"

    # Check if powermetrics exists
    try:
        result = subprocess.run(
            ['which', 'powermetrics'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return False, "powermetrics command not found (should be at /usr/bin/powermetrics)"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "Failed to check for powermetrics command"

    # Check if passwordless sudo access is configured
    try:
        result = subprocess.run(
            ['sudo', '-n', 'powermetrics', '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, "powermetrics is available with passwordless sudo access"
        else:
            return False, "powermetrics requires password (run backend/profiling/setup_powermetrics.sh)"
    except subprocess.TimeoutExpired:
        return False, "powermetrics check timed out"
    except FileNotFoundError:
        return False, "sudo command not found"
    except Exception as e:
        return False, f"Unexpected error checking powermetrics: {e}"


def verify_powermetrics_on_startup() -> None:
    """
    Verify powermetrics access on backend startup and log appropriate message.

    This should be called during FastAPI startup to inform users about
    powermetrics availability.
    """
    is_available, message = check_powermetrics_access()

    if is_available:
        logger.info("✓ Energy Profiler: %s", message)
    else:
        logger.warning("⚠ Energy Profiler: %s", message)
        logger.warning(
            "Energy profiling will be limited to timing and layer metrics. "
            "Power monitoring will not be available."
        )
        logger.warning(
            "To enable power monitoring, run: backend/profiling/setup_powermetrics.sh"
        )


def get_powermetrics_status() -> dict:
    """
    Get detailed status information about powermetrics availability.

    Returns:
        dict: Status information including:
            - available (bool): Whether powermetrics is accessible
            - platform (str): Current platform
            - message (str): Status message
    """
    is_available, message = check_powermetrics_access()

    return {
        'available': is_available,
        'platform': sys.platform,
        'message': message,
        'setup_instructions': 'Run backend/profiling/setup_powermetrics.sh' if not is_available else None
    }
