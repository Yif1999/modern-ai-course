import { useCallback, useEffect, useMemo, useState } from "react";
import { BarChart3, GitCompare, RotateCcw, Search, Zap } from "lucide-react";
import { api } from "../api";

const DEFAULT_PROMPT = "甲：你觉得人工智能以后会改变什么？\n乙：";

function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "暂无数据";
  return Number(value).toFixed(digits);
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "暂无数据";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function modelCheckpointName(item) {
  return item?.name || "";
}

function CandidateBars({ candidates, compact = false }) {
  const maxProb = Math.max(...(candidates || []).map((item) => Number(item.probability) || 0), 1e-12);
  if (!candidates?.length) return <div className="empty">暂无数据</div>;
  return (
    <div className={compact ? "candidate-bars compact" : "candidate-bars"}>
      {candidates.map((item) => (
        <div className={`candidate-bar-row ${item.readable === false ? "unreadable" : ""}`} key={`${item.rank}-${item.id}`}>
          <div className="candidate-label">
            <span>#{item.rank}</span>
            <strong title={item.readable === false ? `${item.kind || "unreadable"} · id ${item.id}` : item.raw_decoded || item.display || item.decoded}>
              {item.display || item.decoded || "∅"}
            </strong>
            <small>{item.readable === false ? item.kind || "byte" : item.id}</small>
          </div>
          <div className="candidate-bar-track">
            <div className="candidate-bar-fill" style={{ width: `${Math.max(2, (item.probability / maxProb) * 100)}%` }} />
          </div>
          <code>{pct(item.probability)}</code>
        </div>
      ))}
    </div>
  );
}

function TokenChips({ tokens, limit = 160 }) {
  const rows = (tokens || []).slice(-limit);
  if (!rows.length) return <div className="empty">暂无数据</div>;
  return (
    <div className="probe-token-grid">
      {rows.map((item) => (
        <div className={`probe-token-chip ${item.readable === false ? "unreadable" : ""}`} key={`${item.index}-${item.id}`}>
          <strong title={item.readable === false ? `${item.kind || "unreadable"} · id ${item.id}` : item.raw_decoded || item.display || item.decoded}>
            {item.display || item.decoded || "∅"}
          </strong>
          <small>
            #{item.index} · {item.readable === false ? item.kind || "byte" : `id ${item.id}`}
          </small>
        </div>
      ))}
    </div>
  );
}

function CandidateTable({ candidates }) {
  if (!candidates?.length) return <div className="empty">暂无数据</div>;
  return (
    <div className="probe-table-wrap">
      <table className="probe-table">
        <thead>
          <tr>
            <th>rank</th>
            <th>token</th>
            <th>id</th>
            <th>prob</th>
            <th>logit</th>
            <th>recent repeat</th>
          </tr>
        </thead>
        <tbody>
          {candidates.map((item) => (
            <tr key={`${item.rank}-${item.id}`}>
              <td>{item.rank}</td>
              <td>
                <strong className={item.readable === false ? "unreadable-token-text" : ""}>{item.display || item.decoded || "∅"}</strong>
              </td>
              <td>{item.id}</td>
              <td>{pct(item.probability)}</td>
              <td>{fmt(item.logit, 3)}</td>
              <td>{item.repeats_recent_context ? "yes" : "no"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ProbeLab({ runs }) {
  const [runId, setRunId] = useState("");
  const [checkpoints, setCheckpoints] = useState([]);
  const [checkpointName, setCheckpointName] = useState("");
  const [compareNames, setCompareNames] = useState([]);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [topK, setTopK] = useState(20);
  const [temperature, setTemperature] = useState(1.0);
  const [steps, setSteps] = useState(8);
  const [maxContextTokens, setMaxContextTokens] = useState(1024);
  const [nextTokenData, setNextTokenData] = useState(null);
  const [traceData, setTraceData] = useState(null);
  const [compareData, setCompareData] = useState(null);
  const [tokenizeData, setTokenizeData] = useState(null);
  const [loading, setLoading] = useState("");
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
      setCompareNames([]);
      try {
        const result = await api.checkpoints(runId);
        if (cancelled) return;
        const next = result.checkpoints || [];
        setCheckpoints(next);
        const modelNames = next.map(modelCheckpointName).filter((name) => name.endsWith("_model.safetensors"));
        const preferred =
          modelNames.find((name) => name === "latest_model.safetensors") ||
          modelNames.find((name) => name === "best_val_model.safetensors") ||
          modelNames[0] ||
          "";
        setCheckpointName(preferred);
        setCompareNames(modelNames.slice(0, 3));
        setNextTokenData(null);
        setTraceData(null);
        setCompareData(null);
        setTokenizeData(null);
      } catch (err) {
        if (!cancelled) setError(err.message || String(err));
      }
    }
    loadCheckpoints();
    return () => {
      cancelled = true;
    };
  }, [runId]);

  const payload = useCallback(
    () => ({
      run_id: runId,
      checkpoint_name: checkpointName,
      prompt,
      top_k: Number(topK),
      temperature: Number(temperature),
      max_context_tokens: Number(maxContextTokens),
      temperature_values: [0.3, 0.5, 0.8, 1.0, 1.3, 1.5],
    }),
    [checkpointName, maxContextTokens, prompt, runId, temperature, topK],
  );

  async function runAction(label, action) {
    if (!runId || !checkpointName) {
      setError("请选择 run 和 *_model.safetensors checkpoint。");
      return;
    }
    setLoading(label);
    setError("");
    try {
      await action();
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading("");
    }
  }

  function toggleCompareName(name) {
    setCompareNames((current) => {
      if (current.includes(name)) return current.filter((item) => item !== name);
      return [...current, name].slice(0, 4);
    });
  }

  return (
    <div className="probe-layout">
      <div className="probe-left-column">
        <section className="panel probe-control-panel">
          <div className="panel-header">
            <div>
              <h2>Checkpoint Probe Lab</h2>
              <div className="panel-note">手动加载 checkpoint 做轻量推理，不自动刷新，不自动占用额外内存</div>
            </div>
            <button className="toggle" onClick={() => api.probeUnload().catch(() => {})}>
              <RotateCcw size={15} />
              unload model
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
              Top-K
              <input type="number" min="1" max="100" value={topK} onChange={(event) => setTopK(event.target.value)} />
            </label>
            <label>
              Temperature
              <input
                type="number"
                min="0.1"
                max="3"
                step="0.1"
                value={temperature}
                onChange={(event) => setTemperature(event.target.value)}
              />
            </label>
            <label>
              Trace steps
              <input type="number" min="1" max="64" value={steps} onChange={(event) => setSteps(event.target.value)} />
            </label>
            <label>
              Context tokens
              <input
                type="number"
                min="1"
                max="1024"
                value={maxContextTokens}
                onChange={(event) => setMaxContextTokens(event.target.value)}
              />
            </label>
          </div>

          <label className="probe-prompt-label">
            Prompt
            <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={6} />
          </label>

          <div className="probe-actions">
            <button
              className="toggle on"
              disabled={!!loading}
              onClick={() =>
                runAction("next", async () => {
                  const result = await api.probeNextToken(payload());
                  setNextTokenData(result);
                  setTokenizeData(result);
                })
              }
            >
              <Zap size={15} />
              Run Next Token Probe
            </button>
            <button
              className="toggle"
              disabled={!!loading}
              onClick={() =>
                runAction("trace", async () => {
                  const result = await api.probeGenerationTrace({ ...payload(), steps: Number(steps) });
                  setTraceData(result);
                })
              }
            >
              <Search size={15} />
              Generation Trace
            </button>
            <button
              className="toggle"
              disabled={!!loading}
              onClick={() =>
                runAction("compare", async () => {
                  const result = await api.probeCheckpointCompare({
                    run_id: runId,
                    checkpoint_names: compareNames,
                    prompt,
                    top_k: Math.min(Number(topK), 12),
                    temperature: Number(temperature),
                    max_context_tokens: Number(maxContextTokens),
                  });
                  setCompareData(result);
                })
              }
            >
              <GitCompare size={15} />
              Compare Checkpoints
            </button>
          </div>
          {loading ? <div className="dataset-note">Running {loading} probe...</div> : null}
          {error ? <div className="dataset-note warning">{error}</div> : null}
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Tokenization Preview</h2>
              <div className="panel-note">当前 prompt 的上下文 token；如果超过 context window，会只显示末尾窗口</div>
            </div>
            <button
              className="toggle"
              disabled={!!loading}
              onClick={() =>
                runAction("tokenize", async () => {
                  const result = await api.probeTokenize(payload());
                  setTokenizeData(result);
                })
              }
            >
              tokenize only
            </button>
          </div>
          {tokenizeData ? (
            <>
              <div className="probe-stat-strip">
                <span>tokens {tokenizeData.token_count}</span>
                <span>context {tokenizeData.context_token_count}</span>
                <span>{tokenizeData.context_truncated ? "truncated" : "full context"}</span>
              </div>
              <TokenChips tokens={tokenizeData.tokenization || tokenizeData.tokens} />
            </>
          ) : (
            <div className="empty">点击 tokenize only 或 Run Next Token Probe</div>
          )}
        </section>
      </div>

      <div className="probe-main-column">
        <section className="panel">
        <div className="panel-header">
          <div>
            <h2>Top-K Next Token</h2>
            <div className="panel-note">取最后一个位置 logits，softmax 后展示最可能的下一个 token</div>
          </div>
          <BarChart3 size={18} />
        </div>
        {nextTokenData ? (
          <div className="probe-two-column">
            <div>
              <div className="probe-stat-grid">
                <div>
                  <span>entropy</span>
                  <strong>{fmt(nextTokenData.stats?.entropy, 3)}</strong>
                </div>
                <div>
                  <span>normalized entropy</span>
                  <strong>{pct(nextTokenData.stats?.normalized_entropy)}</strong>
                </div>
                <div>
                  <span>top prob</span>
                  <strong>{pct(nextTokenData.stats?.top_probability)}</strong>
                </div>
                <div>
                  <span>repeat risk</span>
                  <strong>{pct(nextTokenData.stats?.repetition_ratio)}</strong>
                </div>
              </div>
              <CandidateBars candidates={nextTokenData.candidates} />
            </div>
            <CandidateTable candidates={nextTokenData.candidates} />
          </div>
        ) : (
          <div className="empty">暂无数据</div>
        )}
        </section>

        <section className="panel">
        <div className="panel-header">
          <div>
            <h2>Temperature Comparison</h2>
            <div className="panel-note">同一组 logits，在不同 temperature 下概率分布会变尖锐或变平</div>
          </div>
        </div>
        {nextTokenData?.temperature_views?.length ? (
          <div className="temperature-grid">
            {nextTokenData.temperature_views.map((view) => (
              <div className="temperature-card" key={view.temperature}>
                <div className="temperature-card-title">
                  <strong>T={view.temperature}</strong>
                  <span>entropy {fmt(view.stats?.entropy, 2)}</span>
                </div>
                <CandidateBars candidates={view.candidates.slice(0, 6)} compact />
              </div>
            ))}
          </div>
        ) : (
          <div className="empty">先运行 Next Token Probe</div>
        )}
        </section>

        <section className="panel">
        <div className="panel-header">
          <div>
            <h2>Generation Trace</h2>
            <div className="panel-note">逐步生成，每一步展示 top-k 候选和最终选择的 token</div>
          </div>
        </div>
        {traceData ? (
          <>
            <pre className="probe-generated-text">{traceData.generated_text}</pre>
            <div className="trace-list">
              {traceData.trace.map((step) => (
                <div className="trace-step" key={step.step}>
                  <div className="trace-step-title">
                    <strong>step {step.step}</strong>
                    <span>
                      selected <b>{step.selected.display || step.selected.decoded}</b> · {pct(step.selected.probability)}
                    </span>
                    <span>entropy {fmt(step.stats?.entropy, 2)}</span>
                  </div>
                  <CandidateBars candidates={step.candidates.slice(0, 8)} compact />
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="empty">点击 Generation Trace</div>
        )}
        </section>

        <section className="panel">
        <div className="panel-header">
          <div>
            <h2>Checkpoint Comparison</h2>
            <div className="panel-note">同一 prompt 对比多个 checkpoint 的下一个 token 分布</div>
          </div>
        </div>
        <div className="checkpoint-checkbox-grid">
          {modelCheckpoints.map((item) => (
            <label key={item.name}>
              <input
                type="checkbox"
                checked={compareNames.includes(item.name)}
                onChange={() => toggleCompareName(item.name)}
              />
              {item.name}
            </label>
          ))}
        </div>
        {compareData ? (
          <div className="compare-grid">
            {compareData.comparisons.map((item) => (
              <div className="compare-card" key={item.checkpoint_name}>
                <h3>{item.checkpoint_name}</h3>
                <div className="probe-stat-strip">
                  <span>entropy {fmt(item.stats?.entropy, 2)}</span>
                  <span>top {pct(item.stats?.top_probability)}</span>
                  <span>repeat {pct(item.stats?.repetition_ratio)}</span>
                </div>
                <CandidateBars candidates={item.candidates.slice(0, 8)} compact />
              </div>
            ))}
          </div>
        ) : (
          <div className="empty">选择 checkpoint 后点击 Compare Checkpoints</div>
        )}
        </section>
      </div>
    </div>
  );
}
