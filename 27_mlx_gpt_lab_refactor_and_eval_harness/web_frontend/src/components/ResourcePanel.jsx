function fmt(value, digits = 1, suffix = "") {
  if (value === null || value === undefined || value === "") return "暂无数据";
  if (typeof value === "number") return `${value.toFixed(digits)}${suffix}`;
  return String(value);
}

function clampPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 0;
  return Math.max(0, Math.min(100, Number(value)));
}

function Bar({ label, value, detail, tone = "default" }) {
  const percent = clampPercent(value);
  return (
    <div className="resource-bar">
      <div className="resource-bar-header">
        <span>{label}</span>
        <strong>{value === null || value === undefined ? "暂无数据" : `${percent.toFixed(1)}%`}</strong>
      </div>
      <div className="resource-track">
        <div className={`resource-fill ${tone}`} style={{ width: `${percent}%` }} />
      </div>
      {detail && <div className="resource-detail">{detail}</div>}
    </div>
  );
}

function getPerf(status, metrics) {
  return status?.performance || status?.telemetry || metrics?.performance || {};
}

export default function ResourcePanel({ resources, status, metrics }) {
  const perf = getPerf(status, metrics);
  const cpu = resources?.cpu || {};
  const memory = resources?.memory || {};
  const processes = resources?.training_processes || [];
  const mainProcess = processes[0] || {};

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>System Resources</h2>
        <span className="panel-note">ordinary user mode</span>
      </div>

      <div className="resource-layout">
        <div className="resource-bars">
          <Bar
            label="CPU load"
            value={cpu.load_1m_percent}
            detail={`load: ${fmt(cpu.load_1m, 2)} / cores: ${cpu.cpu_count || "暂无数据"}`}
            tone="cyan"
          />
          <Bar
            label="training process CPU"
            value={resources?.training_cpu_percent}
            detail={mainProcess.pid ? `pid ${mainProcess.pid} · ${mainProcess.elapsed}` : "未检测到训练进程"}
            tone="green"
          />
          <Bar
            label="system memory"
            value={memory.used_percent}
            detail={`${fmt(memory.used_gb, 2, " GB")} used / ${fmt(memory.total_gb, 2, " GB")} total`}
            tone="amber"
          />
        </div>

        <div className="resource-cards">
          <div className="resource-card">
            <span>MLX peak</span>
            <strong>{fmt(perf.mlx_peak_memory_gb, 2, " GB")}</strong>
          </div>
          <div className="resource-card">
            <span>MLX active</span>
            <strong>{fmt(perf.mlx_active_memory_gb, 2, " GB")}</strong>
          </div>
          <div className="resource-card">
            <span>process RSS</span>
            <strong>{fmt(mainProcess.rss_gb ?? perf.process_rss_gb, 2, " GB")}</strong>
          </div>
          <div className="resource-card">
            <span>tokens/sec</span>
            <strong>{fmt(status?.tokens_per_second ?? perf.tokens_per_second, 1)}</strong>
          </div>
        </div>
      </div>

      <p className="panel-footnote">
        GPU 利用率、风扇转速和温度通常需要 sudo / SMC / powermetrics 权限；当前主面板只显示稳定可读的普通权限指标。
      </p>
    </section>
  );
}
