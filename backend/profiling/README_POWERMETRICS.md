# Energy Profiler - powermetrics Setup

The Energy Profiler uses Apple's `powermetrics` utility to measure CPU, GPU, ANE (Apple Neural Engine), and DRAM power consumption during model inference.

## Requirements

- macOS (Apple Silicon recommended, especially M4 Max)
- `powermetrics` utility (included with macOS)
- Passwordless sudo access to `/usr/bin/powermetrics`

## Quick Setup (Automated)

Run the setup script to configure passwordless sudo access:

```bash
cd backend/profiling
./setup_powermetrics.sh
```

The script will:
1. Create a sudoers entry for your user
2. Validate the configuration
3. Test powermetrics access

## Manual Setup

If you prefer to configure manually or the automated script fails:

### Step 1: Create sudoers file

```bash
# Replace YOUR_USERNAME with your actual username
sudo visudo -f /etc/sudoers.d/powermetrics-YOUR_USERNAME
```

### Step 2: Add the following line

```
YOUR_USERNAME ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
```

Save and exit (Ctrl+X, then Y in nano, or :wq in vim).

### Step 3: Verify the configuration

```bash
# This should run without asking for a password
sudo -n powermetrics --help
```

If it works without prompting for a password, you're all set!

### Step 4: Logout and login

If the test in Step 3 fails, logout and login again for the changes to take effect.

## Troubleshooting

### "command not found: powermetrics"

`powermetrics` should be available on macOS by default at `/usr/bin/powermetrics`. If it's missing, ensure you're running macOS and have the latest system updates.

### "syntax error" when editing sudoers

Make sure you:
- Replace `YOUR_USERNAME` with your actual username (run `whoami` to check)
- Don't have any typos in the path `/usr/bin/powermetrics`
- Have the correct format: `username ALL=(ALL) NOPASSWD: /path/to/powermetrics`

### Still prompts for password

Try:
1. Logout and login again
2. Check file permissions: `ls -l /etc/sudoers.d/powermetrics-*` should show `0440`
3. Validate sudoers syntax: `sudo visudo -c`

### Permission denied when running profiler

Ensure the backend is calling powermetrics with sudo:
```python
subprocess.Popen(['sudo', 'powermetrics', ...])
```

## Security Considerations

This configuration grants passwordless sudo access **only** to the `powermetrics` command. This is:

- **Specific**: Only `/usr/bin/powermetrics` is allowed
- **Read-only**: powermetrics only reads system metrics
- **Safe**: powermetrics cannot modify system state
- **Local**: Only affects your user account

If you have security concerns, you can:
1. Remove the sudoers entry after profiling: `sudo rm /etc/sudoers.d/powermetrics-YOUR_USERNAME`
2. Use the profiler only when needed and disable it afterward
3. Run the profiler in a dedicated development environment

## Verification

To verify the Energy Profiler can access powermetrics:

```bash
cd backend
python -c "from profiling.power_monitor import PowerMonitor; print('✓ OK' if PowerMonitor.is_available() else '✗ FAILED')"
```

This should print `✓ OK` if everything is configured correctly.

## Disabling powermetrics Access

To remove the sudoers entry:

```bash
# Replace YOUR_USERNAME with your actual username
sudo rm /etc/sudoers.d/powermetrics-YOUR_USERNAME
```

The Energy Profiler will still function but will display a warning that power monitoring is unavailable. Other profiling features (timing, layer metrics) will continue to work.
