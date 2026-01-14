# Energy Profiler Setup and Usage Guide

The Energy Profiler provides comprehensive power and energy analysis for transformer model inference on Apple Silicon. It measures CPU, GPU, ANE (Apple Neural Engine), and DRAM power consumption with per-token, per-layer, and per-component granularity.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation and Setup](#installation-and-setup)
3. [Configuring powermetrics Access](#configuring-powermetrics-access)
4. [Starting the Energy Profiler](#starting-the-energy-profiler)
5. [Using the Frontend Interface](#using-the-frontend-interface)
6. [Understanding the Metrics](#understanding-the-metrics)
7. [Analyzing Results](#analyzing-results)
8. [Interpreting Energy Data](#interpreting-energy-data)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

## System Requirements

### Hardware
- **Apple Silicon Mac** (M1, M2, M3, M4, or later)
- M4 Max with 128GB RAM recommended for best results
- Sufficient storage for profiling database (varies by usage)

### Software
- **macOS** (any version with Apple Silicon support)
- **Python 3.9+** with PyTorch 2.0+
- **Node.js 18+** for the frontend
- **powermetrics** utility (included with macOS)

### Model Requirements
- HuggingFace transformer models
- Supported architectures: Llama, Mistral, Phi, Qwen
- Models should be compatible with PyTorch MPS backend

## Installation and Setup

### Step 1: Install Dependencies

**Backend dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

The profiling system requires:
- `pynvml` - Power monitoring helpers
- `psutil` - System resource monitoring
- `torch` - PyTorch for model inference
- `transformers` - HuggingFace transformers library

**Frontend dependencies:**
```bash
cd ml-dashboard
npm install
```

Required packages include:
- `d3` - For treemap visualizations
- `d3-sankey` - For Sankey flow diagrams

### Step 2: Verify Installation

Test that the profiling system is correctly installed:

```bash
cd backend
python -c "from profiling import power_monitor, layer_profiler, pipeline_profiler; print('✓ Profiling modules loaded successfully')"
```

### Step 3: Initialize Database

The profiling database is automatically created on first run. To manually initialize:

```bash
cd backend
python -c "from profiling.database import ProfileDatabase; db = ProfileDatabase(); print('✓ Database initialized')"
```

This creates `backend/profiling.db` with the complete schema for storing profiling data.

## Configuring powermetrics Access

The Energy Profiler uses Apple's `powermetrics` utility to measure power consumption. This requires passwordless sudo access.

### Automated Setup (Recommended)

Run the setup script:

```bash
cd backend/profiling
./setup_powermetrics.sh
```

The script will:
1. Create a sudoers entry for your user
2. Validate the configuration
3. Test powermetrics access

### Manual Setup

If the automated script fails, follow these steps:

1. **Create sudoers file:**
   ```bash
   sudo visudo -f /etc/sudoers.d/powermetrics-$(whoami)
   ```

2. **Add this line (replace YOUR_USERNAME):**
   ```
   YOUR_USERNAME ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
   ```

3. **Save and exit** (Ctrl+X, then Y)

4. **Verify the configuration:**
   ```bash
   sudo -n powermetrics --help
   ```

   This should run without asking for a password.

5. **Test with the profiler:**
   ```bash
   cd backend
   python -c "from profiling.power_monitor import PowerMonitor; print('✓ OK' if PowerMonitor.is_available() else '✗ FAILED')"
   ```

### Security Note

This configuration grants passwordless sudo access **only** to the `powermetrics` command. This is safe because:
- Only `/usr/bin/powermetrics` is allowed (specific path)
- powermetrics is read-only (cannot modify system state)
- Only applies to your user account (not system-wide)

See [backend/profiling/README_POWERMETRICS.md](../backend/profiling/README_POWERMETRICS.md) for detailed security information.

### Fallback Mode

If you cannot configure powermetrics access, the Energy Profiler will still work in **fallback mode**:
- Power and energy metrics will be unavailable
- Timing, layer metrics, and activation statistics will still be collected
- A warning message will be displayed in the frontend

## Starting the Energy Profiler

### Quick Start (Both Frontend and Backend)

```bash
cd ml-dashboard
npm run dev:all
```

This starts:
- **Frontend** at `http://localhost:3000`
- **Backend API** at `http://localhost:8000`

### Separate Terminals

**Backend:**
```bash
cd backend
python main.py
```

**Frontend:**
```bash
cd ml-dashboard
npm run dev
```

### Verify Profiler is Running

1. Open your browser to `http://localhost:3000`
2. Navigate to the **Energy Profiler** tab
3. You should see the profiling controls panel

If you see a warning about powermetrics access, follow the [Configuring powermetrics Access](#configuring-powermetrics-access) section.

## Using the Frontend Interface

The Energy Profiler frontend has three main views:

### 1. Real-Time View (Live Profiling)

The **Live** view shows profiling data as it's collected during inference.

**Starting a Profiling Run:**

1. **Select a Model**
   - Choose from available models in the dropdown
   - Models are automatically discovered from your trained-models directory

2. **Enter a Prompt**
   - Type the prompt you want to profile
   - Longer prompts will show prefill phase characteristics
   - Shorter prompts with many output tokens will emphasize decode phase

3. **Configure Profiling Depth**
   - **Module Level**: Profiles at the component level (q_proj, k_proj, MLP, etc.)
     - Lower overhead (~5-10%)
     - Faster execution
     - Recommended for production use

   - **Deep Level**: Profiles individual tensor operations
     - Higher overhead (~15-25%)
     - Captures attention entropy, softmax timing, etc.
     - Best for detailed analysis and optimization

4. **Add Tags (Optional)**
   - Tag runs for easy filtering later
   - Examples: "baseline", "optimized", "experiment-1"

5. **Click "Start Profiling"**
   - The model will load (if not already loaded)
   - Inference will begin with profiling enabled
   - Real-time data will stream to the dashboard

**During Profiling:**

- **Power Timeline Chart**: Shows CPU, GPU, ANE, and DRAM power in real-time
  - Blue line: CPU power
  - Green line: GPU power
  - Orange line: ANE power
  - Purple line: DRAM power
  - Red line: Total power

- **Live Layer Heatmap**: Updates after each token is generated
  - Y-axis: Layer number (0 to N)
  - X-axis: Component type (attention, MLP, layernorm)
  - Color intensity: Energy consumption (darker = more energy)

- **Token Stream**: Displays generated tokens as they appear
  - Tokens are color-coded by energy consumption
  - Hover over a token to see detailed timing

- **Current Operation Indicator**: Shows what's currently being profiled
  - Phase: pre_inference → prefill → decode → post_inference
  - Section: Current operation (embedding, attention, etc.)
  - Layer: Current transformer layer being processed

### 2. Analysis View (Post-Inference Exploration)

The **Analysis** view provides deep exploration of completed profiling runs.

**Components:**

1. **Pipeline Timeline**
   - Horizontal bar showing inference phases
   - Width proportional to duration
   - Color indicates energy consumption
   - Click to expand section details

2. **Metric Selector**
   - Time (ms): How long each component took
   - Energy (mJ): Energy consumed by each component
   - Power (mW): Average power during component execution
   - Activation Mean: Average activation magnitude
   - Activation Max: Maximum activation value
   - Sparsity: Percentage of near-zero activations
   - Attention Entropy: Attention distribution uniformity (deep mode only)

3. **Heatmap Chart**
   - Select a metric to visualize across layers and components
   - Identify hotspots (high energy/time components)
   - Click a cell to drill down into deep operation metrics

4. **Token Slider**
   - Scrub through the decode phase token by token
   - See how layer metrics change per token
   - Use play/pause for animated progression

5. **Treemap View**
   - Hierarchical visualization of energy breakdown
   - Size represents energy consumption
   - Click to zoom into layers → components → operations

6. **Sankey Diagram**
   - Shows energy flow from input through layers to output
   - Width represents energy consumption
   - Highlights attention vs MLP split

### 3. History Browser (Past Runs and Comparison)

The **History** view lets you browse, search, and compare past profiling runs.

**Features:**

1. **Run List**
   - Shows all profiling runs with summary metrics
   - Columns: Date, Model, Prompt (preview), Duration, Energy
   - Click a run to view details

2. **Search and Filters**
   - **Search by prompt**: Find runs containing specific text
   - **Filter by date**: Select date range
   - **Filter by model**: Show runs for specific models only
   - **Filter by tags**: Show runs with specific tags
   - **Sort options**: Date, duration, energy, tokens/second

3. **Run Detail View**
   - Full profiling data for selected run
   - All charts and visualizations from Analysis view
   - Export buttons (JSON, CSV)
   - Delete button (with confirmation)

4. **Comparison Mode**
   - Select 2-4 runs to compare side-by-side
   - Normalized metrics for fair comparison
   - Overlay charts showing differences
   - Statistical comparison (mean, std dev, p-values)

## Understanding the Metrics

### Power and Energy Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| **CPU Power** | mW | Power consumed by CPU cores |
| **GPU Power** | mW | Power consumed by GPU (Metal) |
| **ANE Power** | mW | Power consumed by Apple Neural Engine |
| **DRAM Power** | mW | Power consumed by unified memory |
| **Total Power** | mW | Sum of all components |
| **Energy** | mJ | Power × Time (1 mJ = 1 mW × 1 ms) |
| **Joules per Token** | J/token | Energy per generated token |

### Inference Phase Metrics

| Phase | Description | Key Metrics |
|-------|-------------|-------------|
| **Pre-Inference** | Setup before model execution | Tokenization, tensor transfer |
| **Prefill** | Process entire prompt at once | Input tokens, KV cache initialization |
| **Decode** | Generate tokens one at a time | Output tokens, tokens/second |
| **Post-Inference** | Cleanup after generation | Detokenization, memory cleanup |

### Layer and Component Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Duration** | Time spent in component | Identify slow components |
| **Energy** | Energy consumed by component | Find energy hotspots |
| **Activation Mean** | Average magnitude of activations | Detect dead neurons |
| **Activation Std** | Variability of activations | Assess layer stability |
| **Activation Max** | Maximum activation value | Detect outliers or overflow risk |
| **Sparsity** | Percentage of near-zero activations | Pruning opportunity analysis |

### Deep Operation Metrics (Deep Mode Only)

| Metric | Description |
|--------|-------------|
| **Attention Entropy** | Uniformity of attention distribution (low = focused, high = dispersed) |
| **Attention Sparsity** | Percentage of near-zero attention weights |
| **MLP Activation Kill Ratio** | Percentage of negative inputs killed by activation function |
| **LayerNorm Variance Ratio** | Input variance / output variance |

## Analyzing Results

### Finding Energy Hotspots

1. **Check Pipeline Timeline**
   - Identify which phase uses most energy (usually decode)
   - Look for unexpectedly long phases

2. **Use Treemap Visualization**
   - Size = energy consumption
   - Drill down to find specific layers or components

3. **Analyze Heatmap**
   - Dark red cells = high energy
   - Common hotspots: attention o_proj, MLP down_proj

4. **Compare Across Runs**
   - Same prompt, different models → model efficiency
   - Same model, different prompts → prompt complexity

### Optimizing for Energy Efficiency

**Key Findings from Research:**

1. **Output tokens cost ~11× more than input tokens**
   - Prefill is parallel (efficient)
   - Decode is sequential (expensive)
   - Strategy: Use longer prompts to reduce output length

2. **Model architecture matters more than parameter count**
   - Same parameter count can have order-of-magnitude energy differences
   - Check architectural features: layers, hidden size, attention type

3. **Grouped Query Attention (GQA) is more efficient than Multi-Head (MHA)**
   - Look for models with `num_key_value_heads < num_attention_heads`

4. **Mixture-of-Experts (MoE) uses 2-3× less energy than dense models**
   - Check if model is MoE in architectural features

### Performance vs Efficiency Tradeoffs

Use the **Throughput vs Energy Tradeoff** analysis to find optimal configurations:

1. **Tokens per Joule**: Energy efficiency (higher is better)
2. **Tokens per Second**: Throughput (higher is better)
3. **Energy-Delay Product (EDP)**: Balanced metric (lower is better)
   - EDP = Energy × Duration
   - Optimizes both speed and efficiency

## Interpreting Energy Data

### Typical Energy Patterns

**Normal Patterns:**
- Prefill energy scales linearly with input tokens
- Decode energy scales linearly with output tokens
- GPU dominates power consumption (~60% of total)
- Power spikes during layer execution, drops during memory operations

**Concerning Patterns:**
- Extremely high CPU power (may indicate CPU fallback)
- Constant high ANE power (may indicate unnecessary ANE usage)
- Exponentially increasing energy per token (KV cache saturation)

### Comparing Models

When comparing different models:

1. **Normalize by prompt length**
   - Use "Joules per Token" for fair comparison
   - Account for different input/output lengths

2. **Consider architectural differences**
   - Larger models naturally use more energy
   - Check energy per million parameters

3. **Look at prefill vs decode ratio**
   - Higher ratio = more efficient for long outputs
   - Lower ratio = better for short responses

### Cost and Carbon Estimation

The profiler estimates electricity cost and CO2 emissions:

**Cost Calculation:**
```
Cost (USD) = (Energy in kWh) × (Electricity price per kWh)
Default: $0.12 per kWh
```

**Carbon Calculation:**
```
CO2 (grams) = (Energy in kWh) × (Carbon intensity g/kWh)
Default: 400g for US average grid
```

Regional carbon intensity presets:
- California: ~200g/kWh (clean grid)
- Texas: ~500g/kWh (fossil-heavy grid)
- EU Average: ~300g/kWh
- US Average: ~400g/kWh

## Troubleshooting

### powermetrics Issues

**"Permission denied" when starting profiler:**
- Run the powermetrics setup script: `./backend/profiling/setup_powermetrics.sh`
- Verify with: `sudo -n powermetrics --help`
- If still failing, logout and login again

**"powermetrics not found":**
- Ensure you're on macOS (powermetrics is macOS-only)
- Check that powermetrics exists: `ls /usr/bin/powermetrics`

**Power metrics show zero or incorrect values:**
- Verify powermetrics is running: `ps aux | grep powermetrics`
- Check backend logs for parsing errors
- Try running manually: `sudo powermetrics --samplers cpu_power,gpu_power -i 100`

### Model Loading Issues

**Model fails to load:**
- Ensure model exists in the trained-models directory
- Check that model has config.json and .safetensors files
- Verify sufficient memory (models require RAM for loading)

**"MPS not available":**
- Ensure you're on Apple Silicon
- Check PyTorch installation: `python -c "import torch; print(torch.backends.mps.is_available())"`
- Reinstall PyTorch for Apple Silicon if needed

### Frontend Issues

**Charts not rendering:**
- Check browser console for JavaScript errors
- Ensure WebSocket connection is established
- Verify backend is running on port 8000

**WebSocket disconnected:**
- Check backend is running
- Look for firewall blocking WebSocket connections
- Try refreshing the page

**Data not updating in real-time:**
- Verify WebSocket is connected (check browser console)
- Ensure profiling run is active
- Check backend logs for errors

### Database Issues

**"Database is locked" error:**
- Close any other processes accessing the database
- Check for orphaned connections: `lsof backend/profiling.db`
- Restart the backend server

**Profiling runs not appearing in history:**
- Verify run completed successfully (check logs)
- Refresh the history view
- Check database integrity: `sqlite3 backend/profiling.db "PRAGMA integrity_check;"`

## Advanced Usage

### Exporting Data

**Export as JSON:**
- Full nested structure with all metrics
- Use for programmatic analysis
- Compatible with Python pandas, R, etc.

```bash
# Via API
curl http://localhost:8000/api/profiling/export/RUN_ID?format=json > profile.json
```

**Export as CSV:**
- Flattened tables for spreadsheet analysis
- Multiple CSV files (runs, power_samples, tokens, layers, etc.)
- Use for Excel, Google Sheets, etc.

```bash
# Via API
curl http://localhost:8000/api/profiling/export/RUN_ID?format=csv > profile.csv
```

### Batch Profiling

To profile multiple prompts or models automatically:

```python
import requests

models = ["model-v1", "model-v2", "model-v3"]
prompts = [
    "Explain quantum computing",
    "Write a short story",
    "Solve this math problem: 2x + 5 = 15"
]

for model in models:
    for prompt in prompts:
        response = requests.post("http://localhost:8000/api/profiling/generate", json={
            "model_name": model,
            "prompt": prompt,
            "profiling_depth": "module",
            "tags": ["batch-experiment"],
            "max_length": 100
        })
        run_id = response.json()["run_id"]
        print(f"Completed: {model} - {prompt[:30]}... (run_id: {run_id})")
```

### Custom Analysis Scripts

Use the API to build custom analysis tools:

```python
import requests
import pandas as pd

# Fetch all runs
response = requests.get("http://localhost:8000/api/profiling/runs")
runs = response.json()

# Convert to DataFrame for analysis
df = pd.DataFrame(runs)

# Find most energy-efficient runs
efficient = df.sort_values("joules_per_token").head(10)

# Calculate statistics
print(f"Average energy per token: {df['joules_per_token'].mean():.3f} J")
print(f"Most efficient model: {efficient.iloc[0]['model_name']}")
```

### Integration with Training Pipeline

Profile models after training to track efficiency improvements:

```python
# After training completes
from profiling.pipeline_profiler import InferencePipelineProfiler

profiler = InferencePipelineProfiler(
    model=trained_model,
    profiling_depth="module"
)

with profiler.run(prompt="Test prompt", tags=["post-training"]) as session:
    output = model.generate(...)

summary = profiler.get_summary()
print(f"Energy per token: {summary['joules_per_token']:.3f} J")
```

### Cleaning Up Old Runs

To manage database size, periodically delete old profiling runs:

```bash
# Via API (delete specific run)
curl -X DELETE http://localhost:8000/api/profiling/run/RUN_ID

# Via Python (delete runs older than 30 days)
import requests
from datetime import datetime, timedelta

cutoff = datetime.now() - timedelta(days=30)
response = requests.get("http://localhost:8000/api/profiling/runs")
runs = response.json()

for run in runs:
    run_date = datetime.fromisoformat(run["created_at"])
    if run_date < cutoff:
        requests.delete(f"http://localhost:8000/api/profiling/run/{run['id']}")
        print(f"Deleted old run: {run['id']}")
```

## Additional Resources

- [powermetrics Setup Guide](../backend/profiling/README_POWERMETRICS.md)
- [PRD Document](../plans/prd.json)
- [Design Document](../docs/plans/2026-01-13-energy-profiler-design.md)
- [Main README](../README.md)

## Support and Feedback

If you encounter issues not covered in this guide:

1. Check the backend logs: `tail -f backend/logs/profiler.log`
2. Check the browser console for frontend errors
3. Verify your setup meets the system requirements
4. Review the troubleshooting section above

For bugs or feature requests, please file an issue in the project repository.
