import { useEffect, useMemo, useState } from "react";
import { Network, RotateCcw, Search } from "lucide-react";
import { api } from "../api";

const DEFAULT_QUERY = "马嘉祺";

function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "暂无数据";
  return Number(value).toFixed(digits);
}

function modelCheckpointName(item) {
  return item?.name || "";
}

function TokenPills({ tokens }) {
  if (!tokens?.length) return <div className="empty">暂无 token</div>;
  return (
    <div className="atlas-token-row">
      {tokens.map((token) => (
        <div className={`probe-token-chip ${token.readable === false ? "unreadable" : ""}`} key={`${token.index}-${token.id}`}>
          <strong title={token.raw_decoded || token.display}>{token.display || token.decoded || "∅"}</strong>
          <small>id {token.id}</small>
        </div>
      ))}
    </div>
  );
}

function SimilarityBars({ neighbors }) {
  if (!neighbors?.length) return <div className="empty">暂无近邻</div>;
  const values = neighbors.map((item) => Number(item.similarity)).filter((value) => Number.isFinite(value));
  const minSimilarity = values.length ? Math.min(...values) : 0;
  const maxSimilarity = values.length ? Math.max(...values) : 1;
  const span = Math.max(maxSimilarity - minSimilarity, 1e-6);
  return (
    <div className="candidate-bars">
      {neighbors.map((item) => {
        const value = Number(item.similarity);
        const width = Math.max(3, ((value - minSimilarity) / span) * 100);
        return (
          <div className={`candidate-bar-row ${item.readable === false ? "unreadable" : ""}`} key={`${item.rank}-${item.id}`}>
            <div className="candidate-label">
              <span>#{item.rank}</span>
              <strong title={item.raw_decoded || item.display}>{item.display || item.decoded || "∅"}</strong>
              <small>{item.id}</small>
            </div>
            <div className="candidate-bar-track">
              <div className="candidate-bar-fill" style={{ width: `${width}%` }} />
            </div>
            <code>{fmt(item.similarity, 3)}</code>
          </div>
        );
      })}
    </div>
  );
}

function AtlasMap({ points }) {
  if (!points?.length) return <div className="empty">暂无邻域图</div>;
  const target = points.find((point) => point.is_target) || points[0];
  const neighbors = points.filter((point) => !point.is_target);
  const distances = neighbors.map((point) => Number(point.distance)).filter((value) => Number.isFinite(value));
  const minDistance = distances.length ? Math.min(...distances) : 0;
  const maxDistance = distances.length ? Math.max(...distances) : 1;
  const span = Math.max(maxDistance - minDistance, 1e-6);

  const layoutPoints = [
    { ...target, sx: 50, sy: 50, radius: 9, label: target.display || target.decoded || "∅" },
    ...neighbors.map((point, index) => {
      const angleFromProjection = Math.atan2(Number(point.y || 0), Number(point.x || 0));
      const angle = Number.isFinite(angleFromProjection) && Math.abs(angleFromProjection) > 0.001
        ? angleFromProjection
        : (index / Math.max(neighbors.length, 1)) * Math.PI * 2;
      const normalizedDistance = (Number(point.distance || 0) - minDistance) / span;
      const radial = 17 + Math.max(0, Math.min(1, normalizedDistance)) * 31;
      const rank = Number(point.rank || index + 1);
      const similarity = Number(point.similarity || 0);
      return {
        ...point,
        sx: 50 + Math.cos(angle) * radial,
        sy: 50 - Math.sin(angle) * radial,
        radius: Math.max(4, 8 - Math.min(rank, 20) * 0.12),
        label: point.display || point.decoded || "∅",
        similarity,
      };
    }),
  ];

  return (
    <div className="atlas-map" role="img" aria-label="token semantic neighborhood map">
      <svg className="atlas-map-svg" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
        <defs>
          <radialGradient id="atlasTargetGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(248, 250, 252, 0.95)" />
            <stop offset="100%" stopColor="rgba(45, 212, 191, 0)" />
          </radialGradient>
        </defs>
        <circle className="atlas-ring ring-one" cx="50" cy="50" r="18" />
        <circle className="atlas-ring ring-two" cx="50" cy="50" r="32" />
        <circle className="atlas-ring ring-three" cx="50" cy="50" r="46" />
        <circle className="atlas-target-glow" cx="50" cy="50" r="18" />
        {layoutPoints
          .filter((point) => !point.is_target)
          .map((point) => {
            const opacity = Math.max(0.14, Math.min(0.78, 0.12 + Number(point.similarity || 0) * 0.62));
            return (
              <line
                className="atlas-edge"
                key={`edge-${point.id}-${point.rank}`}
                x1="50"
                y1="50"
                x2={point.sx}
                y2={point.sy}
                style={{ opacity }}
              />
            );
          })}
      </svg>
      {layoutPoints.map((point, index) => {
        const showLabel = point.is_target || Number(point.rank || index) <= 14;
        const similarity = Number(point.similarity || 0);
        return (
          <div
            className={`atlas-node ${point.is_target ? "target" : ""}`}
            key={`${point.id}-${index}`}
            style={{
              left: `${Math.max(4, Math.min(96, point.sx))}%`,
              top: `${Math.max(5, Math.min(95, point.sy))}%`,
              width: `${point.radius * 2}px`,
              height: `${point.radius * 2}px`,
              background: point.is_target ? undefined : `rgba(45, 212, 191, ${Math.max(0.35, Math.min(0.95, similarity))})`,
            }}
            title={`${point.label} · id ${point.id} · sim ${fmt(point.similarity, 4)} · dist ${fmt(point.distance, 4)}`}
          >
            {showLabel ? (
              <span>
                <strong>{point.label}</strong>
                {!point.is_target ? <small>{fmt(point.similarity, 3)}</small> : null}
              </span>
            ) : null}
          </div>
        );
      })}
      <div className="atlas-map-legend">
        <span>center = target</span>
        <span>closer ring = higher cosine similarity</span>
        <span>line brightness = similarity</span>
      </div>
    </div>
  );
}

export default function TokenAtlas({ runs }) {
  const [runId, setRunId] = useState("");
  const [checkpoints, setCheckpoints] = useState([]);
  const [checkpointName, setCheckpointName] = useState("");
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [space, setSpace] = useState("embedding");
  const [topK, setTopK] = useState(20);
  const [includeSelf, setIncludeSelf] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const runOptions = useMemo(() => runs || [], [runs]);

  useEffect(() => {
    if (!runId && runOptions.length) {
      setRunId(runOptions[0].run_id);
    }
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

  async function runAtlasProbe() {
    if (!runId || !checkpointName) {
      setError("请选择 run 和 *_model.safetensors checkpoint。");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const data = await api.probeTokenNeighborhood({
        run_id: runId,
        checkpoint_name: checkpointName,
        query,
        space,
        top_k: Number(topK),
        include_self: includeSelf,
        prompt: "",
        max_targets: 8,
      });
      setResult(data);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="probe-layout token-atlas-layout">
      <div className="probe-left-column">
        <section className="panel probe-control-panel">
          <div className="panel-header">
            <div>
              <h2>Token Atlas</h2>
              <div className="panel-note">输入文本会先拆成 token；每个 token 单独查向量空间最近邻</div>
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
              Vector space
              <select value={space} onChange={(event) => setSpace(event.target.value)}>
                <option value="embedding">input embedding</option>
                <option value="lm_head">lm_head</option>
              </select>
            </label>
            <label>
              Top-K
              <input type="number" min="1" max="80" value={topK} onChange={(event) => setTopK(event.target.value)} />
            </label>
          </div>

          <label className="probe-prompt-label">
            Token text or id
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="马嘉祺 / id:190467 / #42" />
          </label>

          <div className="probe-actions">
            <button className="toggle on" disabled={loading} onClick={runAtlasProbe}>
              <Search size={15} />
              Analyze Tokens
            </button>
            <button className={`toggle ${includeSelf ? "on" : ""}`} disabled={loading} onClick={() => setIncludeSelf((value) => !value)}>
              include self
            </button>
          </div>
          {loading ? <div className="dataset-note">Analyzing token neighborhoods...</div> : null}
          {error ? <div className="dataset-note warning">{error}</div> : null}
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Resolved Tokens</h2>
              <div className="panel-note">每个拆分 token 都会生成独立的近邻排行；最多分析前 8 个 token</div>
            </div>
            <Network size={18} />
          </div>
          {result ? (
            <>
              <div className="probe-stat-strip">
                <span>{result.space}</span>
                <span>{result.weight_tying ? "weight tied" : "untied head"}</span>
                <span>top {result.top_k}</span>
              </div>
              <TokenPills tokens={result.tokenization} />
              <div className="dataset-note">{result.notes?.space}</div>
            </>
          ) : (
            <div className="empty">点击 Analyze Tokens</div>
          )}
        </section>
      </div>

      <div className="probe-main-column">
        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Nearest Token Neighbors</h2>
              <div className="panel-note">每张图和表格都只对应一个目标 token；rank 按 cosine similarity 从高到低排序</div>
            </div>
          </div>
          {result?.targets?.length ? (
            <div className="atlas-target-list">
              {result.targets.map((target) => (
                <div className="atlas-target-card" key={target.id}>
                  <div className="atlas-target-title">
                    <div>
                      <strong>{target.display || target.decoded || "∅"}</strong>
                      <span>id {target.id}</span>
                    </div>
                    <span>{target.neighbors?.length || 0} neighbors</span>
                  </div>
                  <AtlasMap points={target.plot_points} />
                  <div className="atlas-neighbor-grid">
                    <SimilarityBars neighbors={target.neighbors} />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty">暂无数据</div>
          )}
        </section>

      </div>
    </div>
  );
}
