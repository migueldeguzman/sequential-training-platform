# Batch Size Energy Analysis Experiment

## Objective
Quantify the energy consumption and efficiency across different batch sizes for transformer inference on Apple Silicon M4 Max.

## Experimental Setup
- Model: [To be selected, representative of current research]
- Hardware: Apple Silicon M4 Max
- Profiling Tool: SDB Energy Profiler (InferencePipelineProfiler)

## Batch Sizes to Test
- 1 (baseline)
- 2
- 4
- 8
- 16
- 32
- 64
- 128
- 256 (upper limit)

## Metrics to Capture
- Total Energy (mJ)
- Average Power (mW)
- Energy per Token (mJ/token)
- Inference Latency (ms)
- Throughput (tokens/second)

## Experimental Protocol
1. Use same input prompt across all batch sizes
2. Run 5 iterations per batch size
3. Randomize batch size order to minimize systematic bias
4. Cool down between experiments (30-second idle period)

## Analysis Goals
- Identify optimal batch size for energy efficiency
- Understand non-linear scaling of energy with batch size
- Compare with TokenPowerBench (Niu et al. 2025) findings

## Notes
- Potential sources of variation:
  * Input sequence length
  * Model architecture
  * Specific workload characteristics