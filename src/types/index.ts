// Dataset types
export interface Dataset {
  name: string;
  jsonFileCount: number;
  textFileExists: boolean;
  textFilePath?: string;
  pairCount?: number;
}

// Text dataset (standalone .text files)
export interface TextDataset {
  name: string;
  filePath: string;
  fileSize: string;
  sampleCount: number;
  modifiedAt: string;
  datasetType: "text";
}

// Training configuration
export interface TrainingConfig {
  datasets: string[];
  epochs: number;
  learningRate: number;
  sampleSize: number;
  batchMultiplier: number;
  gradientAccumulation: number;
  formatStyle: "chat" | "simple" | "instruction" | "plain";
  trainingMode: "sequential" | "single";
}

// Training status
export interface TrainingStatus {
  isRunning: boolean;
  currentStep: number;
  totalSteps: number;
  currentEpoch: number;
  totalEpochs: number;
  currentDataset: string;
  loss: number;
  progress: number;
  startTime?: string;
  estimatedTimeRemaining?: string;
}

// Log entry
export interface LogEntry {
  timestamp: string;
  level: "info" | "warning" | "error" | "success";
  message: string;
}

// Model checkpoint
export interface ModelCheckpoint {
  name: string;
  path: string;
  createdAt: string;
  datasetsTrained: string[];
  config: Partial<TrainingConfig>;
  size: string;
}

// Conversion job
export interface ConversionJob {
  datasetName: string;
  pairCount: number | null;
  formatStyle: string;
  status: "pending" | "running" | "completed" | "failed";
  outputPath?: string;
}

// Settings
export interface Settings {
  jsonInputDir: string;
  textOutputDir: string;
  textDatasetsDir: string;
  modelOutputDir: string;
  baseModelPath: string;
  pipelineScript: string;
}

// Training history
export interface TrainingHistoryEntry {
  jobId: string;
  startTime: string;
  endTime: string | null;
  status: "completed" | "failed" | "stopped";
  datasets: string[];
  config: Partial<TrainingConfig>;
  finalLoss: number | null;
  logFile: string;
}

export interface TrainingLogDetail extends TrainingHistoryEntry {
  logs: LogEntry[];
}

// Directory browser
export interface DirectoryContents {
  path: string;
  isDir: boolean;
  parent: string;
  contents: { name: string; isDir: boolean; path: string }[];
}

// Inference/Testing types
export interface InferenceConfig {
  modelPath: string;
  temperature: number;
  topK: number;
  topP: number;
  maxLength: number;
  noRepeatNgramSize: number;
  doSample: boolean;
}

export interface InferenceRequest {
  prompt: string;
  config: InferenceConfig;
  repeatCount?: number; // For loop mode
}

export interface InferenceResult {
  id: string;
  prompt: string;
  response: string;
  generationIndex: number;
  timestamp: string;
  config: InferenceConfig;
}

export interface BatchInferenceRequest {
  prompts: string[];
  config: InferenceConfig;
}

export interface ExportableQAPair {
  question: string;
  answer: string;
}

// API Response types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// Energy Profiling types

export interface PowerSample {
  timestamp: number; // Relative to inference start (ms)
  cpu_power_mw: number;
  gpu_power_mw: number;
  ane_power_mw: number;
  dram_power_mw: number;
  total_power_mw: number;
  phase?: string; // idle/pre_inference/prefill/decode/post_inference
}

export interface PipelineSection {
  id: number;
  run_id: string;
  phase: string; // pre_inference/prefill/decode/post_inference
  section_name: string;
  start_time: number; // ms
  end_time: number; // ms
  duration_ms: number;
  energy_mj: number;
}

export interface ComponentMetrics {
  id: number;
  layer_metric_id: number;
  component_name: string; // q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, etc.
  duration_ms: number;
  activation_mean: number;
  activation_std: number;
  activation_max: number;
  sparsity: number;
  deep_operations?: DeepOperationMetrics[];
}

export interface LayerMetrics {
  id: number;
  token_id: number;
  layer_index: number;
  total_duration_ms: number;
  energy_mj: number;
  components: ComponentMetrics[];
}

export interface DeepOperationMetrics {
  id: number;
  component_metric_id: number;
  operation_name: string; // qk_matmul, scale, mask, softmax, value_matmul, etc.
  duration_ms: number;
  attention_entropy?: number;
  max_attention_weight?: number;
  attention_sparsity?: number;
  activation_kill_ratio?: number; // For MLP operations
  variance_ratio?: number; // For LayerNorm
}

export interface TokenMetrics {
  id: number;
  run_id: string;
  token_position: number;
  token_text: string;
  phase: string; // prefill/decode
  start_time: number; // ms
  end_time: number; // ms
  duration_ms: number;
  energy_mj: number;
  power_snapshot_mw: number;
  edp?: number; // Energy-Delay Product (energy_mj × duration_ms)
  layers: LayerMetrics[];
}

export interface ProfilingRun {
  id: string;
  timestamp: string;
  model_name: string;
  model_size_mb: number;
  prompt: string;
  response: string;
  total_duration_ms: number;
  total_energy_mj: number;
  input_tokens: number;
  output_tokens: number;
  // Aliases for consistency with database fields
  input_token_count?: number;
  output_token_count?: number;
  token_count?: number;
  tokens_per_second?: number;
  profiling_depth: string; // module/deep
  inference_engine?: string; // transformers, mlx, vllm, etc.
  tags?: string[];
  experiment_name?: string;
  baseline_power_mw?: number;
  peak_power_mw?: number;
  peak_power_cpu_mw?: number;
  peak_power_gpu_mw?: number;
  peak_power_ane_mw?: number;
  peak_power_dram_mw?: number;
  batch_size?: number;

  // Model architectural features
  total_params?: number;
  num_layers?: number;
  hidden_size?: number;
  intermediate_size?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  attention_mechanism?: string; // MHA/GQA/MQA
  is_moe?: boolean;
  precision?: string; // FP32, FP16, BF16, FP8, INT8, INT4, MIXED
  quantization_method?: string; // gptq, awq, gguf, bitsandbytes, null

  // Calculated metrics
  prefill_energy_mj?: number;
  decode_energy_mj?: number;
  energy_per_input_token_mj?: number;
  energy_per_output_token_mj?: number;
  input_output_energy_ratio?: number;
  joules_per_token?: number;
  joules_per_input_token?: number;
  joules_per_output_token?: number;
  energy_per_million_params?: number;
  tokens_per_joule?: number;
  energy_delay_product?: number;
  // Energy-Delay Product (EDP) metrics
  edp?: number; // Total EDP (total_energy_mj × total_duration_ms)
  edp_per_token?: number; // EDP normalized by token count
  prefill_edp?: number; // EDP for prefill phase only
  decode_edp?: number; // EDP for decode phase only
  cost_usd?: number;
  co2_grams?: number;

  // Related data
  power_samples?: PowerSample[];
  pipeline_sections?: PipelineSection[];
  tokens?: TokenMetrics[];
  summary?: ProfilingRunSummary;
}

export interface ProfilingRunSummary {
  run_id: string;
  total_duration_ms: number;
  total_energy_mj: number;
  phase_breakdown: {
    pre_inference: { duration_ms: number; energy_mj: number };
    prefill: { duration_ms: number; energy_mj: number };
    decode: { duration_ms: number; energy_mj: number };
    post_inference: { duration_ms: number; energy_mj: number };
  };
  phase_power_breakdown?: {
    phase: string;
    sample_count: number;
    avg_power_mw: number;
    peak_power_mw: number;
    avg_cpu_power_mw: number;
    avg_gpu_power_mw: number;
    avg_ane_power_mw: number;
    avg_dram_power_mw: number;
  }[];
  average_layer_metrics: {
    layer_index: number;
    avg_duration_ms: number;
    avg_energy_mj: number;
  }[];
  average_component_metrics: {
    component_name: string;
    avg_duration_ms: number;
  }[];
  hottest_components: {
    component_name: string;
    layer_index: number;
    duration_ms: number;
  }[];
  efficiency_metrics: {
    // EP-076: Comprehensive efficiency metrics
    total_energy_per_token_mj: number; // mJ per token (all tokens)
    prefill_energy_per_token_mj: number; // mJ per input token
    decode_energy_per_token_mj: number; // mJ per output token
    energy_per_million_params_mj: number | null; // mJ per million parameters (requires model features)
    tokens_per_joule: number; // Efficiency score (higher is better)
    power_utilization_percentage: number; // Actual power vs TDP %
    avg_power_mw: number; // Average power during active inference
    joules_per_token: number; // J per token (standardized TokenPowerBench metric)
    joules_per_input_token: number; // J per input token
    joules_per_output_token: number; // J per output token
  };
  // EP-088: Energy-Delay Product metrics
  edp_metrics?: {
    edp: number; // Total EDP (total_energy_mj × total_duration_ms)
    edp_per_token: number; // EDP normalized by token count
    prefill_edp: number; // Prefill phase EDP (prefill_energy_mj × prefill_duration_ms)
    decode_edp: number; // Decode phase EDP (decode_energy_mj × decode_duration_ms)
    prefill_duration_ms: number; // Duration of prefill phase
    decode_duration_ms: number; // Duration of decode phase
  };
  token_energy_breakdown?: {
    input_energy_mj: number;
    output_energy_mj: number;
    input_token_count: number;
    output_token_count: number;
    energy_per_input_token_mj: number;
    energy_per_output_token_mj: number;
    output_to_input_energy_ratio: number;
  };
  component_energy_breakdown?: {
    run_id: string;
    cpu_energy_mj: number;
    gpu_energy_mj: number;
    ane_energy_mj: number;
    dram_energy_mj: number;
    total_energy_mj: number;
    cpu_energy_percentage: number;
    gpu_energy_percentage: number;
    ane_energy_percentage: number;
    dram_energy_percentage: number;
    avg_cpu_power_mw: number;
    avg_gpu_power_mw: number;
    avg_ane_power_mw: number;
    avg_dram_power_mw: number;
    peak_cpu_power_mw: number;
    peak_gpu_power_mw: number;
    peak_ane_power_mw: number;
    peak_dram_power_mw: number;
    sample_count: number;
  };
  component_energy_by_phase?: {
    phase: string;
    cpu_energy_mj: number;
    gpu_energy_mj: number;
    ane_energy_mj: number;
    dram_energy_mj: number;
    total_energy_mj: number;
  }[];
  // EP-089: Cost and carbon estimation metrics
  cost_carbon_metrics?: {
    cost_usd: number; // Total electricity cost in USD
    co2_grams: number; // Total CO2 emissions in grams
    co2_kg: number; // Total CO2 emissions in kg (for convenience)
    cost_per_token_usd: number; // Cost per token
    co2_per_token_grams: number; // CO2 per token
    electricity_price_per_kwh: number; // Electricity price used ($/kWh)
    carbon_intensity_g_per_kwh: number; // Carbon intensity used (g/kWh)
    equivalent_car_miles: number; // Equivalent car miles for CO2 emissions
  };
}

// WebSocket message types

export type ProfilingMessageType =
  | 'power_sample'
  | 'section_start'
  | 'section_end'
  | 'token_complete'
  | 'layer_metrics'
  | 'component_metrics'
  | 'inference_complete'
  | 'model_loading';

export interface BaseProfilingMessage {
  type: ProfilingMessageType;
  timestamp: number; // ms since inference start
}

export interface PowerSampleMessage extends BaseProfilingMessage {
  type: 'power_sample';
  data: PowerSample;
}

export interface SectionStartMessage extends BaseProfilingMessage {
  type: 'section_start';
  data: {
    phase: string;
    section_name: string;
  };
}

export interface SectionEndMessage extends BaseProfilingMessage {
  type: 'section_end';
  data: {
    phase: string;
    section_name: string;
    duration_ms: number;
    energy_mj: number;
  };
}

export interface TokenCompleteMessage extends BaseProfilingMessage {
  type: 'token_complete';
  token_position: number;
  token_text: string;
  duration_ms: number;
  energy_mj: number;
  power_snapshot_mw: number;
  layer_metrics_summary: {
    total_duration_ms: number;
    total_energy_mj: number;
  };
  layer_metrics?: LayerMetrics[];  // Full layer-by-layer data for LiveLayerHeatmap
}

export interface LayerMetricsMessage extends BaseProfilingMessage {
  type: 'layer_metrics';
  token_position: number;
  layer_index: number;
  metrics: Omit<LayerMetrics, 'id' | 'token_id' | 'components'>;
}

export interface ComponentMetricsMessage extends BaseProfilingMessage {
  type: 'component_metrics';
  token_position: number;
  layer_index: number;
  component_name: string;
  metrics: Omit<ComponentMetrics, 'id' | 'layer_metric_id'>;
}

export interface InferenceCompleteMessage extends BaseProfilingMessage {
  type: 'inference_complete';
  run_id: string;
  total_duration_ms: number;
  total_energy_mj: number;
  token_count: number;
  tokens_per_second: number;
  summary: {
    prefill_duration_ms: number;
    prefill_energy_mj: number;
    decode_duration_ms: number;
    decode_energy_mj: number;
    joules_per_token: number;
  };
}

export interface ModelLoadingMessage extends BaseProfilingMessage {
  type: 'model_loading';
  data: {
    status: 'loading' | 'complete' | 'error';
    model_name: string;
    model_path: string;
    message: string;
  };
}

export type ProfilingMessage =
  | PowerSampleMessage
  | SectionStartMessage
  | SectionEndMessage
  | TokenCompleteMessage
  | LayerMetricsMessage
  | ComponentMetricsMessage
  | InferenceCompleteMessage
  | ModelLoadingMessage;

// Profiling API request types

export interface ProfiledGenerateRequest {
  prompt: string;
  model_path?: string;
  profiling_depth?: 'module' | 'deep';
  tags?: string[];
  experiment_name?: string;
  temperature?: number;
  top_p?: number;
  max_length?: number;
  batch_size?: number;
  device?: 'auto' | 'cpu' | 'cuda' | 'mps';
}

export interface ProfilingRunsFilter {
  model?: string;
  date_from?: string;
  date_to?: string;
  tags?: string[];
  experiment?: string;
  inference_engine?: string;
  limit?: number;
  offset?: number;
  sort_by?: 'date' | 'duration' | 'energy' | 'efficiency' | 'joules_per_token';
  sort_order?: 'asc' | 'desc';
}

// Energy Prediction types

export interface EnergyPrediction {
  predicted_total_energy_mj: number;
  predicted_prefill_energy_mj: number;
  predicted_decode_energy_mj: number;
  predicted_energy_per_token_mj: number;
  confidence_interval_95_pct: [number, number];
  model_accuracy_r2: number;
  features_used: string[];
  prediction_notes: string | null;
}

export interface EnergyPredictionRequest {
  model_name: string;
  input_tokens: number;
  output_tokens: number;
  batch_size?: number;
}

// EP-092: Energy Scaling Analysis types

export interface EnergyScalingDataPoint {
  run_id: string;
  model_name: string;
  total_params: number;
  total_params_millions: number;
  total_energy_mj: number;
  energy_per_million_params: number;
  input_tokens: number | null;
  output_tokens: number | null;
  joules_per_token: number;
}

export interface PowerLawFit {
  coefficient_a: number;
  exponent_b: number;
  formula: string;
  r_squared: number | null;
  interpretation: string;
  note?: string;
  error?: string;
  message?: string;
}

export interface ScalingEfficiency {
  smallest_model: {
    name: string;
    params_millions: number;
    energy_per_million_params: number;
  };
  largest_model: {
    name: string;
    params_millions: number;
    energy_per_million_params: number;
  };
  efficiency_gain_pct: number;
  conclusion: string;
}

export interface EnergyScalingAnalysis {
  scaling_data: EnergyScalingDataPoint[];
  power_law_fit: PowerLawFit | null;
  scaling_efficiency: ScalingEfficiency | null;
  statistics: {
    model_count: number;
    total_runs: number;
    param_range_millions?: [number, number];
    energy_range_mj?: [number, number];
    message?: string;
  };
}

// EP-096: Throughput vs Energy Tradeoff Analysis types

export interface ThroughputEnergyDataPoint {
  run_id: string;
  model_name: string;
  throughput_tokens_per_second: number;
  energy_per_token_mj: number;
  tokens_per_joule: number;
  total_energy_mj: number;
  total_tokens: number;
  duration_ms: number;
  batch_size: number | null;
  is_pareto_optimal: boolean;
}

export interface ParetoFrontierPoint {
  run_id: string;
  throughput_tokens_per_second: number;
  energy_per_token_mj: number;
  tokens_per_joule: number;
}

export interface KneePoint {
  run_id: string;
  throughput_tokens_per_second: number;
  energy_per_token_mj: number;
  tokens_per_joule: number;
  interpretation: string;
}

export interface ThroughputEnergyStatistics {
  total_runs: number;
  unique_models: number;
  throughput_range: [number, number];
  energy_per_token_range: [number, number];
  best_throughput: {
    run_id: string;
    model_name: string;
    throughput_tokens_per_second: number;
    energy_per_token_mj: number;
  } | null;
  best_efficiency: {
    run_id: string;
    model_name: string;
    throughput_tokens_per_second: number;
    tokens_per_joule: number;
  } | null;
}

export interface ThroughputEnergyTradeoff {
  data_points: ThroughputEnergyDataPoint[];
  pareto_frontier: ParetoFrontierPoint[];
  knee_point: KneePoint | null;
  statistics: ThroughputEnergyStatistics;
}

// EP-081: Quantization Energy Comparison types

export interface QuantizationRunData {
  run_id: string;
  model_name: string;
  precision: string;
  quantization_method: string | null;
  energy_per_token_mj: number;
  tokens_per_second: number;
  total_energy_mj: number;
  token_count: number;
}

export interface EnergySavings {
  absolute_mj: number;
  percent: number;
}

export interface QuantizationComparison {
  precision_levels: string[];
  runs_by_precision: {
    [precision: string]: QuantizationRunData[];
  };
  average_energy_per_token: {
    [precision: string]: number;
  };
  energy_savings: {
    [comparison: string]: EnergySavings;
  };
  throughput: {
    [precision: string]: number;
  };
  notes: string[];
}
