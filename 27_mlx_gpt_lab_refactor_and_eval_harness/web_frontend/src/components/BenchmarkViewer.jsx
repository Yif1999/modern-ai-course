function pick(obj, keys) {
  for (const key of keys) {
    if (obj && obj[key] !== undefined && obj[key] !== null) return obj[key];
  }
  return null;
}

function fmt(value) {
  if (value === null || value === undefined) return "暂无数据";
  return typeof value === "number" ? value.toFixed(3) : String(value);
}

export default function BenchmarkViewer({ benchmarkData }) {
  const benchmark = benchmarkData?.benchmark || {};
  const metrics = benchmarkData?.metrics || {};
  const tokensPerSecond = pick(benchmark, ["tokens_per_second"]) ?? pick(metrics, ["tokens_per_second"]);
  const elapsed = pick(benchmark, ["elapsed_sec"]) ?? pick(metrics, ["elapsed_sec"]);
  const paramCount = pick(benchmark, ["parameter_count"]) ?? pick(metrics, ["parameter_count"]);

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Benchmark</h2>
      </div>
      <div className="metrics-grid compact">
        <div className="metric">
          <span>tokens/sec</span>
          <strong>{fmt(tokensPerSecond)}</strong>
        </div>
        <div className="metric">
          <span>elapsed</span>
          <strong>{fmt(elapsed)}</strong>
        </div>
        <div className="metric">
          <span>parameters</span>
          <strong>{fmt(paramCount)}</strong>
        </div>
      </div>
      {benchmarkData ? <pre className="code-block small">{JSON.stringify(benchmarkData, null, 2)}</pre> : <div className="empty">暂无数据</div>}
    </section>
  );
}
