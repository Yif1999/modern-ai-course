import { useEffect, useMemo, useState } from "react";
import { Crosshair, RotateCcw, Search } from "lucide-react";
import { api } from "../api";

const DEFAULT_PROMPT = "人工智能正在改变我们的生活，";

function modelCheckpointName(item) {
  return item?.name || "";
}

function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "暂无数据";
  return Number(value).toFixed(digits);
}

function tokenLabel(token) {
  return token?.display || token?.decoded || "∅";
}

function AttentionHeatmap({ data }) {
  const weights = data?.weights || [];
  const tokens = data?.tokens || [];
  if (!weights.length || !tokens.length) return <div className="empty">暂无 attention map</div>;

  const size = weights.length;
  const cell = size <= 4 ? 70 : size <= 8 ? 48 : Math.max(9, Math.min(20, Math.floor(720 / Math.max(size, 1))));
  const labelWidth = size <= 4 ? 128 : 96;
  const topLabelHeight = size <= 4 ? 88 : 76;
  const chartSize = size * cell;
  const width = labelWidth + chartSize + 12;
  const height = topLabelHeight + chartSize + 36;
  const renderWidth = Math.min(920, Math.max(width, size <= 4 ? 360 : size <= 8 ? 520 : 760));

  return (
    <div className="attention-heatmap-wrap">
      <div className="attention-heatmap-inner">
        <svg
          className="attention-heatmap-svg"
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label="attention heatmap"
          style={{ width: `${renderWidth}px` }}
        >
          <g transform={`translate(${labelWidth}, ${topLabelHeight})`}>
            {weights.map((row, rowIndex) =>
              row.map((value, colIndex) => {
                const visible = colIndex <= rowIndex;
                const opacity = visible ? Math.max(0.03, Math.min(1, Number(value) * 8)) : 0.02;
                return (
                  <rect
                    key={`${rowIndex}-${colIndex}`}
                    x={colIndex * cell}
                    y={rowIndex * cell}
                    width={cell - 1}
                    height={cell - 1}
                    fill={visible ? "#5eead4" : "#223044"}
                    opacity={opacity}
                  >
                    <title>
                      {`q ${rowIndex} ${tokenLabel(tokens[rowIndex])} -> k ${colIndex} ${tokenLabel(tokens[colIndex])}: ${fmt(value, 6)}`}
                    </title>
                  </rect>
                );
              }),
            )}
            <rect x="0" y="0" width={chartSize} height={chartSize} fill="none" stroke="#2b394b" strokeWidth="1" />
            <line x1="0" y1="0" x2={chartSize} y2={chartSize} stroke="#f59e0b" strokeOpacity="0.35" strokeWidth="1" />
          </g>

          {tokens.map((token, index) => {
            const label = tokenLabel(token).slice(0, 8);
            const x = labelWidth + index * cell + cell / 2;
            const y = topLabelHeight - 8;
            return (
              <text
                key={`top-${index}-${token.id}`}
                x={x}
                y={y}
                textAnchor="end"
                transform={`rotate(-55 ${x} ${y})`}
                className="attention-axis-label"
              >
                {label}
              </text>
            );
          })}

          {tokens.map((token, index) => (
            <text
              key={`left-${index}-${token.id}`}
              x={labelWidth - 8}
              y={topLabelHeight + index * cell + cell * 0.68}
              textAnchor="end"
              className="attention-axis-label"
            >
              {index} {tokenLabel(token).slice(0, 7)}
            </text>
          ))}

          <text x={labelWidth} y={height - 8} className="attention-caption">
            row = query position, column = key position, brighter = stronger attention, upper triangle is causal-masked
          </text>
        </svg>
      </div>
    </div>
  );
}

function HeadSummary({ data, selectedHead, onSelectHead }) {
  const heads = data?.head_summaries || [];
  if (!heads.length) return <div className="empty">暂无 head summary</div>;
  return (
    <div className="attention-head-grid">
      {heads.map((head) => (
        <button
          className={`attention-head-card ${Number(head.head) === Number(selectedHead) ? "selected" : ""}`}
          key={head.head}
          onClick={() => onSelectHead(head.head)}
        >
          <strong>head {head.head}</strong>
          <span>kv {head.kv_head}</span>
          <small>last entropy {fmt(head.last_token_entropy, 2)}</small>
        </button>
      ))}
    </div>
  );
}

function LastTokenTop({ summary }) {
  const rows = summary?.last_token_top || [];
  if (!rows.length) return <div className="empty">暂无数据</div>;
  const maxWeight = Math.max(...rows.map((row) => Number(row.weight) || 0), 1e-12);
  return (
    <div className="candidate-bars">
      {rows.map((row, index) => (
        <div className={`candidate-bar-row ${row.token?.readable === false ? "unreadable" : ""}`} key={`${row.position}-${index}`}>
          <div className="candidate-label">
            <span>#{row.position}</span>
            <strong title={row.token?.raw_decoded || tokenLabel(row.token)}>{tokenLabel(row.token)}</strong>
            <small>{row.token?.id}</small>
          </div>
          <div className="candidate-bar-track">
            <div className="candidate-bar-fill" style={{ width: `${Math.max(2, (Number(row.weight) / maxWeight) * 100)}%` }} />
          </div>
          <code>{fmt(row.weight, 4)}</code>
        </div>
      ))}
    </div>
  );
}

export default function AttentionInspector({ runs }) {
  const [runId, setRunId] = useState("");
  const [checkpoints, setCheckpoints] = useState([]);
  const [checkpointName, setCheckpointName] = useState("");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [maxContextTokens, setMaxContextTokens] = useState(64);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const runOptions = useMemo(() => runs || [], [runs]);

  useEffect(() => {
    if (!runId && runOptions.length) setRunId(runOptions[0].run_id);
  }, [runId, runOptions]);

  const modelCheckpoints = useMemo(
    () => checkpoints.filter((item) => modelCheckpointName(item).endsWith("_model.safetensors")),
    [checkpoints],
  );

  useEffect(() => {
    let cancelled = false;
    async function loadCheckpoints() {
      if (!runId) return;
      setCheckpoints([]);
      setCheckpointName("");
      setResult(null);
      try {
        const data = await api.checkpoints(runId);
        if (cancelled) return;
        const next = data.checkpoints || [];
        setCheckpoints(next);
        const modelNames = next.map(modelCheckpointName).filter((name) => name.endsWith("_model.safetensors"));
        const preferred =
          modelNames.find((name) => name === "latest_model.safetensors") ||
          modelNames.find((name) => name === "best_val_model.safetensors") ||
          modelNames[0] ||
          "";
        setCheckpointName(preferred);
      } catch (err) {
        if (!cancelled) setError(err.message || String(err));
      }
    }
    loadCheckpoints();
    return () => {
      cancelled = true;
    };
  }, [runId]);

  async function runAttentionProbe(nextHead = head) {
    if (!runId || !checkpointName) {
      setError("请选择 run 和 *_model.safetensors checkpoint。");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const data = await api.probeAttention({
        run_id: runId,
        checkpoint_name: checkpointName,
        prompt,
        layer: Number(layer),
        head: Number(nextHead),
        max_context_tokens: Number(maxContextTokens),
      });
      setLayer(data.layer);
      setHead(data.head);
      setResult(data);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  function selectHead(nextHead) {
    setHead(nextHead);
    runAttentionProbe(nextHead);
  }

  return (
    <div className="probe-layout attention-layout">
      <div className="probe-left-column">
        <section className="panel probe-control-panel">
          <div className="panel-header">
            <div>
              <h2>Attention Head Inspector</h2>
              <div className="panel-note">手动计算指定 layer/head 的 causal attention，不影响训练主循环</div>
            </div>
            <button className="toggle" onClick={() => api.probeUnload().catch(() => {})}>
              <RotateCcw size={15} />
              unload
            </button>
          </div>

          <div className="probe-form-grid">
            <label>
              Run
              <select value={runId} onChange={(event) => setRunId(event.target.value)}>
                {runOptions.map((run) => (
                  <option value={run.run_id} key={run.run_id}>
                    {run.run_id}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Checkpoint
              <select value={checkpointName} onChange={(event) => setCheckpointName(event.target.value)}>
                {modelCheckpoints.length === 0 ? <option value="">暂无 checkpoint</option> : null}
                {modelCheckpoints.map((item) => (
                  <option value={item.name} key={item.name}>
                    {item.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Layer
              <input type="number" min="0" max="99" value={layer} onChange={(event) => setLayer(event.target.value)} />
            </label>
            <label>
              Head
              <input type="number" min="0" max="99" value={head} onChange={(event) => setHead(event.target.value)} />
            </label>
            <label>
              Context tokens
              <input
                type="number"
                min="2"
                max="192"
                value={maxContextTokens}
                onChange={(event) => setMaxContextTokens(event.target.value)}
              />
            </label>
          </div>

          <label className="probe-prompt-label">
            Prompt
            <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={7} />
          </label>

          <div className="probe-actions">
            <button className="toggle on" disabled={loading} onClick={() => runAttentionProbe()}>
              <Search size={15} />
              Run Attention Probe
            </button>
          </div>
          {loading ? <div className="dataset-note">Computing attention weights...</div> : null}
          {error ? <div className="dataset-note warning">{error}</div> : null}
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Heads</h2>
              <div className="panel-note">GQA 中多个 Q head 会共享同一个 K/V head</div>
            </div>
            <Crosshair size={18} />
          </div>
          {result ? (
            <>
              <div className="probe-stat-strip">
                <span>{result.num_layers} layers</span>
                <span>{result.num_q_heads} q heads</span>
                <span>{result.num_kv_heads} kv heads</span>
                <span>head dim {result.head_dim}</span>
              </div>
              <HeadSummary data={result} selectedHead={head} onSelectHead={selectHead} />
            </>
          ) : (
            <div className="empty">点击 Run Attention Probe</div>
          )}
        </section>
      </div>

      <div className="probe-main-column">
        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Attention Heatmap</h2>
              <div className="panel-note">
                layer {result?.layer ?? layer} · head {result?.head ?? head} · kv head {result?.kv_head ?? "暂无数据"}
              </div>
            </div>
          </div>
          {result ? <AttentionHeatmap data={result} /> : <div className="empty">暂无数据</div>}
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Last Token Attention</h2>
              <div className="panel-note">最后一个 query token 最关注的历史 token；适合观察当前 next-token 判断依赖了什么</div>
            </div>
          </div>
          {result ? (
            <div className="probe-two-column">
              <LastTokenTop summary={result.selected_head_summary} />
              <div>
                <div className="probe-stat-grid">
                  <div>
                    <span>selected layer</span>
                    <strong>{result.layer}</strong>
                  </div>
                  <div>
                    <span>selected head</span>
                    <strong>{result.head}</strong>
                  </div>
                  <div>
                    <span>kv head</span>
                    <strong>{result.kv_head}</strong>
                  </div>
                  <div>
                    <span>context tokens</span>
                    <strong>{result.context_token_count}</strong>
                  </div>
                </div>
                <div className="dataset-note">
                  Heatmap 是 checkpoint 分析路径手算出来的 attention weights；训练时仍使用 fast attention。
                </div>
              </div>
            </div>
          ) : (
            <div className="empty">暂无数据</div>
          )}
        </section>
      </div>
    </div>
  );
}
