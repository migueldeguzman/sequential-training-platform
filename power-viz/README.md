# Phase-Tagged Power Visualization

## Overview
Interactive dashboard for visualizing phase-tagged power consumption during AI inference.

## Features
- Parse phase-tagged power sample data
- Interactive multi-phase selection
- Real-time power consumption tracing

## Requirements
- Python 3.9+
- See requirements.txt for dependencies

## Usage
1. Prepare power sample data in CSV format
2. Run `python power_viz.py`
3. Access dashboard at `http://localhost:8050`

## Data Format
CSV should include columns:
- `timestamp`
- `power_watts`
- `phase`