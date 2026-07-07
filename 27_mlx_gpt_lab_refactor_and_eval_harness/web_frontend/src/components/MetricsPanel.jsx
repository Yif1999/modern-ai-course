function fmt(value, digits = 4) {
  if (value === null || value === undefined || value === "") return "暂无数据";
  if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(digits);
  return String(value);
}

function compact(value, digits = 2) {
  if (value === null || value === undefined || value === "") return "暂无数据";
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  const abs = Math.abs(number);
  if (abs >= 1e12) return `${(number / 1e12).toFixed(digits)}T`;
  if (abs >= 1e9) return `${(number / 1e9).toFixed(digits)}B`;
  if (abs >= 1e6) return `${(number / 1e6).toFixed(digits)}M`;
  if (abs >= 1e3) return `${(number / 1e3).toFixed(digits)}K`;
  return Number.isInteger(number) ? number.toLocaleString() : number.toFixed(digits);
}

function fmtAge(value) {
  if (!value) return "暂无数据";
  const stamp = typeof value === "number" ? value * 1000 : Date.parse(value);
  if (!Number.isFinite(stamp)) return "暂无数据";
  const age = Math.max(0, Math.floor((Date.now() - stamp) / 1000));
  if (age < 60) return `${age}s ago`;
  const minutes = Math.floor(age / 60);
  const seconds = age % 60;
  return `${minutes}m ${seconds}s ago`;
}

function fmtSeconds(value) {
  if (value === null || value === undefined || value === "") return "暂无数据";
  const total = Number(value);
  if (!Number.isFinite(total)) return "暂无数据";
  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = Math.floor(total % 60);
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

function fmtLearningRate(value) {
  if (value === null || value === undefined || value === "") return "暂无数据";
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  if (number === 0) return "0";
  if (Math.abs(number) < 0.01) return number.toExponential(2);
  return number.toFixed(4);
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function MetricsPanel({ status, metrics, checkpoints }) {
  const rawMetrics = metrics || {};
  const rawStatus = status || {};
  const perf = rawStatus.performance || rawStatus.telemetry || rawMetrics.performance || {};
  const parameterCount = rawStatus.parameter_count ?? rawMetrics.parameter_count;
  const estimatedTflops =
    rawStatus.estimated_training_tflops ?? rawMetrics.estimated_training_tflops ?? perf.estimated_training_tflops;
  const totalPflops = rawStatus.estimated_total_pflops ?? rawMetrics.estimated_total_pflops ?? perf.estimated_total_pflops;
  const tokensPerParameter =
    rawStatus.tokens_per_parameter ?? rawMetrics.tokens_per_parameter ?? perf.tokens_per_parameter;
  const progress = rawStatus.progress_percent ?? rawMetrics.progress_percent ?? perf.progress_percent;
  const eta = rawStatus.eta_sec ?? rawMetrics.eta_sec ?? perf.eta_sec;
  const stepTime = perf.step_time_ms;
  const batchSize =
    rawStatus.micro_batch_size ?? rawStatus.batch_size ?? rawMetrics.micro_batch_size ?? rawMetrics.batch_size;
  const accumSteps =
    rawStatus.gradient_accumulation_steps ??
    rawStatus.grad_accum_steps ??
    rawMetrics.gradient_accumulation_steps ??
    rawMetrics.grad_accum_steps;

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Metrics / Performance</h2>
        <span className="panel-note">source: {perf.source || rawStatus.status_source || "status.json"}</span>
      </div>
      <div className="metrics-grid">
        <Metric label="batch size" value={fmt(batchSize, 0)} />
        <Metric label="latest step" value={fmt(rawStatus.step ?? rawMetrics.max_iters, 0)} />
        <Metric label="progress" value={progress === undefined || progress === null ? "暂无数据" : `${fmt(progress, 2)}%`} />
        <Metric label="train loss" value={fmt(rawStatus.train_loss ?? rawMetrics.final_train_loss)} />
        <Metric label="val loss" value={fmt(rawStatus.val_loss ?? rawMetrics.final_val_loss)} />
        <Metric label="best val loss" value={fmt(rawStatus.best_val_loss ?? rawMetrics.best_val_loss)} />
        <Metric label="tokens_seen" value={fmt(rawStatus.tokens_seen ?? rawMetrics.tokens_seen, 0)} />
        <Metric label="tokens/sec" value={fmt(rawStatus.tokens_per_second ?? rawMetrics.tokens_per_second, 2)} />
        <Metric label="step time" value={stepTime === undefined || stepTime === null ? "暂无数据" : `${fmt(stepTime, 2)} ms`} />
        <Metric label="ETA" value={fmtSeconds(eta)} />
        <Metric label="parameters" value={compact(parameterCount)} />
        <Metric label="MLX active memory" value={perf.mlx_active_memory_gb === undefined ? "暂无数据" : `${fmt(perf.mlx_active_memory_gb, 2)} GB`} />
        <Metric label="MLX peak memory" value={perf.mlx_peak_memory_gb === undefined ? "暂无数据" : `${fmt(perf.mlx_peak_memory_gb, 2)} GB`} />
        <Metric label="MLX cache memory" value={perf.mlx_cache_memory_gb === undefined ? "暂无数据" : `${fmt(perf.mlx_cache_memory_gb, 2)} GB`} />
        <Metric label="est TFLOPS" value={fmt(estimatedTflops, 2)} />
        <Metric label="total PFLOPs" value={fmt(totalPflops, 2)} />
        <Metric label="tokens / param" value={fmt(tokensPerParameter, 4)} />
        <Metric label="elapsed time" value={fmt(rawStatus.elapsed_sec ?? rawMetrics.elapsed_sec, 2)} />
        <Metric label="accum steps" value={fmt(accumSteps, 0)} />
        <Metric label="learning rate" value={fmtLearningRate(rawStatus.learning_rate ?? rawMetrics.learning_rate)} />
      </div>
      <p className="panel-footnote">
        这里合并训练状态与轻量性能指标；est TFLOPS / total PFLOPs 使用教学估算：训练 FLOPs 约等于 6 × 参数量 × token 数。
      </p>
    </section>
  );
}
