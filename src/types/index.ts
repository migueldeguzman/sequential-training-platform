// Dataset types
export interface Dataset {
  name: string;
  jsonFileCount: number;
  textFileExists: boolean;
  textFilePath?: string;
  pairCount?: number;
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

// Auth types
export interface User {
  username: string;
  isAuthenticated: boolean;
}

export interface AuthState {
  user: User | null;
  token: string | null;
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
  profiling_depth: string; // module/deep
  tags?: string[];
  experiment_name?: string;
  baseline_power_mw?: number;
  peak_power_mw?: number;
  peak_power_cpu_mw?: number;
  peak_power_gpu_mw?: number;
  peak_power_ane_mw?: number;
  peak_power_dram_mw?: number;
  batch_size?: number;
  inference_engine?: string;

  // Model architectural features
  total_params?: number;
  num_layers?: number;
  hidden_size?: number;
  intermediate_size?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  attention_mechanism?: string; // MHA/GQA/MQA
  is_moe?: boolean;

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
}

// WebSocket message types

export type ProfilingMessageType =
  | 'power_sample'
  | 'section_start'
  | 'section_end'
  | 'token_complete'
  | 'layer_metrics'
  | 'component_metrics'
  | 'inference_complete';

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
  phase: string;
  section_name: string;
}

export interface SectionEndMessage extends BaseProfilingMessage {
  type: 'section_end';
  phase: string;
  section_name: string;
  duration_ms: number;
  energy_mj: number;
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

export type ProfilingMessage =
  | PowerSampleMessage
  | SectionStartMessage
  | SectionEndMessage
  | TokenCompleteMessage
  | LayerMetricsMessage
  | ComponentMetricsMessage
  | InferenceCompleteMessage;

// Profiling API request types

export interface ProfiledGenerateRequest {
  prompt: string;
  model_path?: string;
  profiling_depth?: 'module' | 'deep';
  tags?: string[];
  experiment_name?: string;
  temperature?: number;
  max_length?: number;
}

export interface ProfilingRunsFilter {
  model?: string;
  date_from?: string;
  date_to?: string;
  tags?: string[];
  experiment?: string;
  limit?: number;
  offset?: number;
  sort_by?: 'date' | 'duration' | 'energy' | 'efficiency';
  sort_order?: 'asc' | 'desc';
}
