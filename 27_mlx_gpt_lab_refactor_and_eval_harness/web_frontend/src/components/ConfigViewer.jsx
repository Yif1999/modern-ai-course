export default function ConfigViewer({ config }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Config</h2>
      </div>
      {config ? <pre className="code-block">{JSON.stringify(config, null, 2)}</pre> : <div className="empty">暂无数据</div>}
    </section>
  );
}
