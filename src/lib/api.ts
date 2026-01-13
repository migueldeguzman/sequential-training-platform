import type {
  Dataset,
  TrainingConfig,
  TrainingStatus,
  ModelCheckpoint,
  ConversionJob,
  ApiResponse,
  Settings,
  TrainingHistoryEntry,
  TrainingLogDetail,
  DirectoryContents,
  InferenceConfig,
  InferenceResult,
  ExportableQAPair,
  ProfiledGenerateRequest,
  ProfilingRun,
  ProfilingRunSummary,
  PipelineSection,
  ProfilingRunsFilter,
  PowerSample,
  TokenMetrics,
} from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<ApiResponse<T>> {
  try {
    const token = localStorage.getItem("auth_token");
    const headers: HeadersInit = {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options?.headers,
    };

    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const error = await response.json();
      return { success: false, error: error.detail || "Request failed" };
    }

    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "Network error",
    };
  }
}

// Dataset API
export const datasetApi = {
  list: () => fetchApi<Dataset[]>("/api/datasets"),

  getDetails: (name: string) => fetchApi<Dataset>(`/api/datasets/${name}`),

  convert: (job: ConversionJob) =>
    fetchApi<{ outputPath: string }>("/api/datasets/convert", {
      method: "POST",
      body: JSON.stringify(job),
    }),

  batchConvert: (jobs: ConversionJob[]) =>
    fetchApi<{ outputPaths: string[] }>("/api/datasets/batch-convert", {
      method: "POST",
      body: JSON.stringify({ jobs }),
    }),

  preview: (name: string, limit = 5) =>
    fetchApi<{
      fileName: string;
      filePath: string;
      fileSize: string;
      totalPairs: number;
      previewPairs: string[];
      format: string;
    }>(`/api/datasets/${name}/preview?limit=${limit}`),

  getFormat: (name: string) =>
    fetchApi<{
      datasetName: string;
      jsonFileCount: number;
      formatTemplate: string;
      sampleJson: { question: string; answer: string } | null;
      sampleConverted: string | null;
      formats: Record<string, string>;
    }>(`/api/datasets/${name}/format`),
};

// Files API
export const filesApi = {
  listConverted: () =>
    fetchApi<{
      files: {
        name: string;
        path: string;
        size: string;
        pairCount: number;
        modifiedAt: string;
      }[];
    }>("/api/files/converted"),

  getContent: (path: string, offset = 0, limit = 10) =>
    fetchApi<{
      totalPairs: number;
      offset: number;
      limit: number;
      pairs: string[];
      hasMore: boolean;
    }>(`/api/files/content?path=${encodeURIComponent(path)}&offset=${offset}&limit=${limit}`),
};

// Training API
export const trainingApi = {
  start: (config: TrainingConfig) =>
    fetchApi<{ jobId: string }>("/api/training/start", {
      method: "POST",
      body: JSON.stringify(config),
    }),

  stop: () =>
    fetchApi<{ message: string }>("/api/training/stop", {
      method: "POST",
    }),

  status: () => fetchApi<TrainingStatus>("/api/training/status"),

  logs: (limit = 100) =>
    fetchApi<{ logs: string[] }>(`/api/training/logs?limit=${limit}`),

  // Training history
  history: () =>
    fetchApi<{ history: TrainingHistoryEntry[] }>("/api/training/history"),

  historyDetail: (jobId: string) =>
    fetchApi<TrainingLogDetail>(`/api/training/history/${jobId}`),

  deleteHistory: (jobId: string) =>
    fetchApi<{ message: string }>(`/api/training/history/${jobId}`, {
      method: "DELETE",
    }),
};

// Model API
export const modelApi = {
  list: () => fetchApi<ModelCheckpoint[]>("/api/models"),

  getDetails: (name: string) => fetchApi<ModelCheckpoint>(`/api/models/${name}`),

  delete: (name: string) =>
    fetchApi<{ message: string }>(`/api/models/${name}`, {
      method: "DELETE",
    }),

  download: (name: string) => `${API_BASE}/api/models/${name}/download`,
};

// Settings API
export const settingsApi = {
  get: () => fetchApi<Settings>("/api/settings"),

  update: (settings: Settings) =>
    fetchApi<Settings>("/api/settings", {
      method: "POST",
      body: JSON.stringify(settings),
    }),

  browse: (path: string) =>
    fetchApi<DirectoryContents>(`/api/settings/browse?path=${encodeURIComponent(path)}`, {
      method: "POST",
    }),
};

// Auth API
export const authApi = {
  login: (username: string, password: string) =>
    fetchApi<{ token: string; username: string }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),

  logout: () =>
    fetchApi<{ message: string }>("/api/auth/logout", {
      method: "POST",
    }),

  verify: () => fetchApi<{ valid: boolean; username: string }>("/api/auth/verify"),
};

// Inference API
export const inferenceApi = {
  // Load a model for inference
  loadModel: (modelPath: string) =>
    fetchApi<{ message: string; modelPath: string }>("/api/inference/load", {
      method: "POST",
      body: JSON.stringify({ modelPath }),
    }),

  // Unload current model
  unloadModel: () =>
    fetchApi<{ message: string }>("/api/inference/unload", {
      method: "POST",
    }),

  // Get current model status
  status: () =>
    fetchApi<{ loaded: boolean; modelPath: string | null; deviceInfo: string }>("/api/inference/status"),

  // Single generation
  generate: (prompt: string, config: Partial<InferenceConfig>) =>
    fetchApi<InferenceResult>("/api/inference/generate", {
      method: "POST",
      body: JSON.stringify({ prompt, config }),
    }),

  // Loop generation (same prompt multiple times)
  generateLoop: (prompt: string, repeatCount: number, config: Partial<InferenceConfig>) =>
    fetchApi<{ results: InferenceResult[] }>("/api/inference/generate-loop", {
      method: "POST",
      body: JSON.stringify({ prompt, repeatCount, config }),
    }),

  // Batch generation (multiple different prompts)
  generateBatch: (prompts: string[], config: Partial<InferenceConfig>) =>
    fetchApi<{ results: InferenceResult[] }>("/api/inference/generate-batch", {
      method: "POST",
      body: JSON.stringify({ prompts, config }),
    }),

  // Export results as training data
  exportAsTrainingData: (results: InferenceResult[], format: "json" | "text") =>
    fetchApi<{ data: ExportableQAPair[] | string; filename: string }>("/api/inference/export", {
      method: "POST",
      body: JSON.stringify({ results, format }),
    }),
};

// Profiling API
export const profilingApi = {
  // Start profiled inference with energy tracking
  profiledGenerate: (request: ProfiledGenerateRequest) =>
    fetchApi<{ runId: string; response: string; message: string }>("/api/profiling/generate", {
      method: "POST",
      body: JSON.stringify(request),
    }),

  // Get list of profiling runs with optional filters
  getProfilingRuns: (filter?: ProfilingRunsFilter) => {
    const params = new URLSearchParams();
    if (filter?.model) params.append("model", filter.model);
    if (filter?.date_from) params.append("date_from", filter.date_from);
    if (filter?.date_to) params.append("date_to", filter.date_to);
    if (filter?.tags && filter.tags.length > 0) params.append("tags", filter.tags.join(","));
    if (filter?.experiment) params.append("experiment", filter.experiment);
    if (filter?.limit) params.append("limit", filter.limit.toString());
    if (filter?.offset) params.append("offset", filter.offset.toString());
    if (filter?.sort_by) params.append("sort_by", filter.sort_by);

    const queryString = params.toString();
    const endpoint = queryString ? `/api/profiling/runs?${queryString}` : "/api/profiling/runs";

    return fetchApi<{
      runs: ProfilingRun[];
      total: number;
      limit: number;
      offset: number;
    }>(endpoint);
  },

  // Get complete profiling run data with all nested metrics
  getProfilingRun: (id: string) =>
    fetchApi<{
      run: ProfilingRun;
      power_samples: PowerSample[];
      pipeline_sections: PipelineSection[];
      tokens: TokenMetrics[];
    }>(`/api/profiling/run/${id}`),

  // Get aggregated summary statistics for a profiling run
  getProfilingRunSummary: (id: string) =>
    fetchApi<ProfilingRunSummary>(`/api/profiling/run/${id}/summary`),

  // Get hierarchical pipeline section breakdown
  getProfilingPipeline: (id: string) =>
    fetchApi<{
      run_id: string;
      total_duration_ms: number;
      total_energy_mj: number;
      phases: Array<{
        phase: string;
        sections: PipelineSection[];
        total_duration_ms: number;
        total_energy_mj: number;
        avg_power_mw: number;
        section_count: number;
        duration_percentage: number;
        energy_percentage: number;
      }>;
    }>(`/api/profiling/run/${id}/pipeline`),

  // Export profiling run data (returns blob URL for download)
  exportProfilingRun: (id: string, format: "json" | "csv") => {
    const url = `${API_BASE}/api/profiling/export/${id}?format=${format}`;
    return url;
  },

  // Delete profiling run and all related data
  deleteProfilingRun: (id: string) =>
    fetchApi<{ success: boolean; message: string; run_id: string }>(`/api/profiling/run/${id}`, {
      method: "DELETE",
    }),

  // Get architectural analysis correlating model features with energy consumption
  getArchitecturalAnalysis: (filters?: {
    model_filter?: string;
    min_params?: number;
    max_params?: number;
  }) => {
    const params = new URLSearchParams();
    if (filters?.model_filter) params.append("model_filter", filters.model_filter);
    if (filters?.min_params) params.append("min_params", filters.min_params.toString());
    if (filters?.max_params) params.append("max_params", filters.max_params.toString());

    const queryString = params.toString();
    const endpoint = queryString ? `/api/profiling/architectural-analysis?${queryString}` : "/api/profiling/architectural-analysis";

    return fetchApi<{
      data_points: Array<{
        run_id: string;
        model_name: string;
        num_layers: number;
        hidden_size: number;
        intermediate_size: number;
        num_attention_heads: number;
        attention_mechanism: string;
        total_params: number;
        total_energy_mj: number;
        energy_per_token_mj: number;
        tokens_per_joule: number;
      }>;
      correlations: {
        energy_vs_layers?: {
          coefficient: number | null;
          p_value: number | null;
          interpretation: string;
        };
        energy_vs_hidden_size?: {
          coefficient: number | null;
          p_value: number | null;
          interpretation: string;
        };
        energy_vs_intermediate_size?: {
          coefficient: number | null;
          p_value: number | null;
          interpretation: string;
        };
        energy_vs_total_params?: {
          coefficient: number | null;
          p_value: number | null;
          interpretation: string;
        };
      };
      attention_mechanism_comparison: {
        [mechanism: string]: {
          count: number;
          avg_energy_per_token: number;
          avg_tokens_per_joule: number;
        };
      };
      regression_models: {
        linear_layers: {
          slope: number;
          intercept: number;
          r_squared: number;
          description: string;
        };
        quadratic_hidden_size: {
          coefficient: number;
          intercept: number;
          r_squared: number;
          description: string;
        };
      };
      message?: string;
    }>(endpoint);
  },
};

// Combined API export
export const api = {
  datasets: datasetApi,
  files: filesApi,
  training: trainingApi,
  models: modelApi,
  settings: settingsApi,
  auth: authApi,
  inference: inferenceApi,
  profiling: profilingApi,

  // Direct access to profiling methods for convenience
  profiledGenerate: profilingApi.profiledGenerate,
  getProfilingRuns: profilingApi.getProfilingRuns,
  getProfilingRun: profilingApi.getProfilingRun,
  getProfilingRunSummary: profilingApi.getProfilingRunSummary,
  getProfilingPipeline: profilingApi.getProfilingPipeline,
  exportProfilingRun: profilingApi.exportProfilingRun,
  deleteProfilingRun: profilingApi.deleteProfilingRun,
  getArchitecturalAnalysis: profilingApi.getArchitecturalAnalysis,
};
