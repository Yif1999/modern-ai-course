const API_OVERRIDE_STORAGE_KEY = "mlx_gpt_lab_api_base_url";
const DEFAULT_API_PORT = import.meta.env.VITE_API_PORT || "8765";

function normalizeBaseUrl(value) {
  return String(value || "").trim().replace(/\/+$/, "");
}

function autoApiBaseUrl() {
  if (typeof window === "undefined") {
    return `http://localhost:${DEFAULT_API_PORT}`;
  }
  return `${window.location.protocol}//${window.location.hostname}:${DEFAULT_API_PORT}`;
}

function queryApiBaseUrl() {
  if (typeof window === "undefined") return "";
  const params = new URLSearchParams(window.location.search);
  const value = params.get("api") || params.get("apiBaseUrl");
  if (!value) return "";
  if (value === "auto") {
    window.localStorage.removeItem(API_OVERRIDE_STORAGE_KEY);
    return "";
  }
  const normalized = normalizeBaseUrl(value);
  window.localStorage.setItem(API_OVERRIDE_STORAGE_KEY, normalized);
  return normalized;
}

function storedApiBaseUrl() {
  if (typeof window === "undefined") return "";
  return normalizeBaseUrl(window.localStorage.getItem(API_OVERRIDE_STORAGE_KEY));
}

function computeApiBaseUrl() {
  const envUrl = normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL);
  if (envUrl) return { url: envUrl, source: ".env" };

  const queryUrl = queryApiBaseUrl();
  if (queryUrl) return { url: queryUrl, source: "url" };

  const storedUrl = storedApiBaseUrl();
  if (storedUrl) return { url: storedUrl, source: "saved" };

  return { url: autoApiBaseUrl(), source: "auto" };
}

const apiConfig = computeApiBaseUrl();

export const API_BASE_URL = apiConfig.url;
export const API_BASE_URL_SOURCE = apiConfig.source;

export function saveApiBaseUrlOverride(value) {
  const normalized = normalizeBaseUrl(value);
  if (!normalized) {
    window.localStorage.removeItem(API_OVERRIDE_STORAGE_KEY);
    return;
  }
  window.localStorage.setItem(API_OVERRIDE_STORAGE_KEY, normalized);
}

export function clearApiBaseUrlOverride() {
  window.localStorage.removeItem(API_OVERRIDE_STORAGE_KEY);
}

async function request(path) {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return response.json();
}

async function postRequest(path, payload = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return response.json();
}

function queryString(params = {}) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      search.set(key, value);
    }
  });
  const text = search.toString();
  return text ? `?${text}` : "";
}

export const api = {
  health: () => request("/api/health"),
  systemResources: () => request("/api/system-resources"),
  runs: () => request("/api/runs"),
  run: (runId) => request(`/api/runs/${encodeURIComponent(runId)}`),
  config: (runId) => request(`/api/runs/${encodeURIComponent(runId)}/config`),
  trainingLog: (runId, params) => request(`/api/runs/${encodeURIComponent(runId)}/training-log${queryString(params)}`),
  metrics: (runId) => request(`/api/runs/${encodeURIComponent(runId)}/metrics`),
  status: (runId) => request(`/api/runs/${encodeURIComponent(runId)}/status`),
  samples: (runId) => request(`/api/runs/${encodeURIComponent(runId)}/samples`),
  sample: (runId, sampleName) =>
    request(`/api/runs/${encodeURIComponent(runId)}/samples/${encodeURIComponent(sampleName)}`),
  datasetSamples: (runId, params) =>
    request(`/api/runs/${encodeURIComponent(runId)}/dataset-samples${queryString(params)}`),
  tokenizerVocab: (runId, params) =>
    request(`/api/runs/${encodeURIComponent(runId)}/tokenizer-vocab${queryString(params)}`),
  finalText: (runId) => request(`/api/runs/${encodeURIComponent(runId)}/final-text`),
  evalResults: (runId) => request(`/api/runs/${encodeURIComponent(runId)}/eval-results`),
  benchmark: (runId) => request(`/api/runs/${encodeURIComponent(runId)}/benchmark`),
  checkpoints: (runId) => request(`/api/runs/${encodeURIComponent(runId)}/checkpoints`),
  probeTokenize: (payload) => postRequest("/api/probe/tokenize", payload),
  probeNextToken: (payload) => postRequest("/api/probe/next-token", payload),
  probeGenerationTrace: (payload) => postRequest("/api/probe/generation-trace", payload),
  probeCheckpointCompare: (payload) => postRequest("/api/probe/checkpoint-compare", payload),
  probeTokenNeighborhood: (payload) => postRequest("/api/probe/token-neighborhood", payload),
  probeAttention: (payload) => postRequest("/api/probe/attention", payload),
  probeUnload: () => postRequest("/api/probe/unload", {}),
};
