function fmt(value, digits = 2, suffix = "") {
  if (value === null || value === undefined || value === "") return "暂无数据";
  if (typeof value === "number") {
    const shown = Number.isInteger(value) ? value.toLocaleString() : value.toFixed(digits);
    return `${shown}${suffix}`;
  }
  return `${String(value)}${suffix}`;
}

function seconds(value) {
  if (value === null || value === undefined || value === "") return "暂无数据";
  const total = Number(value);
  if (!Number.isFinite(total)) return "暂无数据";
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = Math.floor(total % 60);
  if (hours > 0) return `${hours}h ${minutes}m ${secs}s`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

function Metric({ label, value, tone = "" }) {
  return (
    <div className={`metric ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function getPerformance(status, metrics) {
  const rawStatus = status || {};
  const rawMetrics = metrics || {};
  return rawStatus.performance || rawStatus.telemetry || rawMetrics.performance || {};
}

export default function PerformancePanel({ status, metrics }) {
  const perf = getPerformance(status, metrics);
  const progress = status?.progress_percent ?? perf.progress_percent;
  const source = perf.source || status?.source || "status.json";

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Performance / Telemetry</h2>
        <span className="panel-note">source: {source}</span>
      </div>
      <div className="metrics-grid performance-grid">
        <Metric label="progress" value={fmt(progress, 2, progress !== undefined ? "%" : "")} />
        <Metric label="ETA" value={seconds(status?.eta_sec ?? perf.eta_sec)} />
        <Metric label="step time" value={fmt(perf.step_time_ms, 2, perf.step_time_ms !== undefined ? " ms" : "")} />
        <Metric label="tokens/sec" value={fmt(status?.tokens_per_second ?? perf.tokens_per_second, 2)} />

        <Metric label="MLX active memory" value={fmt(perf.mlx_active_memory_gb, 2, perf.mlx_active_memory_gb !== undefined ? " GB" : "")} />
        <Metric label="MLX peak memory" value={fmt(perf.mlx_peak_memory_gb, 2, perf.mlx_peak_memory_gb !== undefined ? " GB" : "")} />
        <Metric label="MLX cache memory" value={fmt(perf.mlx_cache_memory_gb, 2, perf.mlx_cache_memory_gb !== undefined ? " GB" : "")} />
        <Metric label="Metal peak memory" value={fmt(perf.metal_peak_memory_gb, 2, perf.metal_peak_memory_gb !== undefined ? " GB" : "")} />
      </div>
      <p className="panel-footnote">
        训练性能优先看 tokens/sec、step time、MLX peak memory。系统 CPU / 内存放在 System Resources 面板。
      </p>
    </section>
  );
}
