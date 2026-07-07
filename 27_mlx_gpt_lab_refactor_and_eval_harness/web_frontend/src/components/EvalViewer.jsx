export default function EvalViewer({ evalData }) {
  const rows = evalData?.rows || [];
  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Eval</h2>
      </div>
      {rows.length === 0 ? (
        <div className="empty">暂无数据</div>
      ) : (
        <div className="eval-list">
          {evalData.metrics && (
            <div className="subtle-block">
              <strong>metrics</strong>
              <pre>{JSON.stringify(evalData.metrics, null, 2)}</pre>
            </div>
          )}
          {rows.map((row, index) => (
            <details key={row.id || index} className="eval-item" open={index === 0}>
              <summary>{row.prompt || row.id || `eval ${index + 1}`}</summary>
              <div className="eval-generation">{row.generated || row.continuation || "暂无数据"}</div>
              <pre>{JSON.stringify(row, null, 2)}</pre>
            </details>
          ))}
        </div>
      )}
    </section>
  );
}
