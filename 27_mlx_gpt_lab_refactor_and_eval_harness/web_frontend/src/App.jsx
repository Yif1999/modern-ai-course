import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, Pause, Play } from "lucide-react";
import { API_BASE_URL, API_BASE_URL_SOURCE, api, clearApiBaseUrlOverride, saveApiBaseUrlOverride } from "./api";
import BenchmarkViewer from "./components/BenchmarkViewer";
import ConfigViewer from "./components/ConfigViewer";
import DatasetPreview from "./components/DatasetPreview";
import EvalViewer from "./components/EvalViewer";
import LossChart from "./components/LossChart";
import MetricsPanel from "./components/MetricsPanel";
import ModelInspector from "./components/ModelInspector";
import RunSelector from "./components/RunSelector";
import SampleViewer from "./components/SampleViewer";
import TokenizerResultPreview from "./components/TokenizerResultPreview";
import "./styles.css";

function useInterval(callback, delay, enabled) {
  useEffect(() => {
    if (!enabled) return undefined;
    const id = setInterval(callback, delay);
    return () => clearInterval(id);
  }, [callback, delay, enabled]);
}

function getErrorMessage(error) {
  if (!error) return "";
  return error.message || String(error);
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [trainingLog, setTrainingLog] = useState([]);
  const [config, setConfig] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [status, setStatus] = useState(null);
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState("");
  const [sampleText, setSampleText] = useState("");
  const [finalText, setFinalText] = useState("");
  const [evalData, setEvalData] = useState(null);
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [checkpoints, setCheckpoints] = useState([]);
  const [datasetPreview, setDatasetPreview] = useState(null);
  const [datasetMode, setDatasetMode] = useState("random");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshIntervalMs, setRefreshIntervalMs] = useState(4000);
  const [view, setView] = useState(() => {
    if (typeof window === "undefined") return "dashboard";
    const requestedView = new URLSearchParams(window.location.search).get("view");
    return requestedView === "probe" || requestedView === "atlas" || requestedView === "attention" || requestedView === "inspector"
      ? "inspector"
      : "dashboard";
  });
  const [inspectorTool, setInspectorTool] = useState(() => {
    if (typeof window === "undefined") return "probe";
    const params = new URLSearchParams(window.location.search);
    const requestedTool = params.get("tool") || params.get("view");
    return ["probe", "attention", "atlas"].includes(requestedTool) ? requestedTool : "probe";
  });
  const [apiBaseInput, setApiBaseInput] = useState(API_BASE_URL);
  const [refreshError, setRefreshError] = useState("");
  const [lastRefreshAt, setLastRefreshAt] = useState(null);
  const liveRefreshInFlight = useRef(false);
  const selectedRunIdRef = useRef("");
  const selectedSampleRef = useRef("");

  const currentRun = useMemo(() => runs.find((run) => run.run_id === selectedRunId), [runs, selectedRunId]);
  const lastRefreshLabel = lastRefreshAt ? new Date(lastRefreshAt).toLocaleTimeString() : "尚未刷新";

  const markRefreshSuccess = useCallback(() => {
    setRefreshError("");
    setLastRefreshAt(Date.now());
  }, []);

  const markRefreshFailure = useCallback((scope, err) => {
    setRefreshError(`${scope}: ${getErrorMessage(err)}`);
  }, []);

  useEffect(() => {
    selectedRunIdRef.current = selectedRunId;
    setTrainingLog([]);
    setConfig(null);
    setMetrics(null);
    setStatus(null);
    setSamples([]);
    setSelectedSample("");
    setSampleText("");
    setFinalText("");
    setEvalData(null);
    setBenchmarkData(null);
    setCheckpoints([]);
    setDatasetPreview(null);
  }, [selectedRunId]);

  useEffect(() => {
    selectedSampleRef.current = selectedSample;
  }, [selectedSample]);

  const loadRuns = useCallback(async () => {
    try {
      const [healthResponse, runsResponse] = await Promise.all([api.health(), api.runs()]);
      setHealth(healthResponse);
      setRuns(runsResponse.runs || []);
      markRefreshSuccess();
      if (!selectedRunId && runsResponse.runs?.length) {
        setSelectedRunId(runsResponse.runs[0].run_id);
      }
    } catch (err) {
      markRefreshFailure("runs refresh failed", err);
    }
  }, [markRefreshFailure, markRefreshSuccess, selectedRunId]);

  const loadRunStaticData = useCallback(async () => {
    if (!selectedRunId) return;
    const runId = selectedRunId;
    const requests = [
      ["training log", api.trainingLog(runId, { heartbeat_points: 900 })],
      ["config", api.config(runId)],
      ["samples", api.samples(runId)],
      ["final text", api.finalText(runId)],
      ["checkpoints", api.checkpoints(runId)],
    ];
    const results = await Promise.allSettled(requests.map(([, request]) => request));
    if (selectedRunIdRef.current !== runId) return;
    const errors = [];
    const valueAt = (index) => {
      const result = results[index];
      if (result.status === "fulfilled") return result.value;
      errors.push(`${requests[index][0]}: ${getErrorMessage(result.reason)}`);
      return null;
    };

    const logRes = valueAt(0);
    const configRes = valueAt(1);
    const samplesRes = valueAt(2);
    const finalTextRes = valueAt(3);
    const ckptRes = valueAt(4);

    setTrainingLog(logRes?.rows || []);
    setConfig(configRes?.config || null);
    setFinalText(finalTextRes?.text || "");
    setCheckpoints(ckptRes?.checkpoints || []);

    if (samplesRes) {
      const nextSamples = samplesRes.samples || [];
      setSamples(nextSamples);
      setSelectedSample((current) =>
        nextSamples.length && !nextSamples.some((sample) => sample.name === current) ? nextSamples[nextSamples.length - 1].name : current,
      );
    }

    if (errors.length) {
      setRefreshError(`部分数据暂不可用，已保留上一次成功数据：${errors.slice(0, 2).join("; ")}`);
    } else {
      markRefreshSuccess();
    }
  }, [markRefreshSuccess, selectedRunId]);

  const loadRunDebugData = useCallback(async () => {
    if (!selectedRunId) return;
    const runId = selectedRunId;
    const requests = [
      ["eval results", api.evalResults(runId)],
      ["benchmark", api.benchmark(runId)],
    ];
    const results = await Promise.allSettled(requests.map(([, request]) => request));
    if (selectedRunIdRef.current !== runId) return;
    const evalRes = results[0].status === "fulfilled" ? results[0].value : null;
    const benchmarkRes = results[1].status === "fulfilled" ? results[1].value : null;
    if (evalRes) setEvalData(evalRes || null);
    if (benchmarkRes) setBenchmarkData(benchmarkRes || null);
  }, [selectedRunId]);

  const loadRunLiveData = useCallback(async () => {
    if (!selectedRunId || liveRefreshInFlight.current) return;
    const runId = selectedRunId;
    liveRefreshInFlight.current = true;
    const requests = [
      ["status", api.status(runId)],
      ["metrics", api.metrics(runId)],
      ["training log", api.trainingLog(runId, { heartbeat_points: 900 })],
      ["samples", api.samples(runId)],
      ["checkpoints", api.checkpoints(runId)],
    ];
    try {
      const results = await Promise.allSettled(requests.map(([, request]) => request));
      if (selectedRunIdRef.current !== runId) return;
      const errors = [];
      const valueAt = (index) => {
        const result = results[index];
        if (result.status === "fulfilled") return result.value;
        errors.push(`${requests[index][0]}: ${getErrorMessage(result.reason)}`);
        return null;
      };

      const statusRes = valueAt(0);
      const metricsRes = valueAt(1);
      const logRes = valueAt(2);
      const samplesRes = valueAt(3);
      const ckptRes = valueAt(4);

      if (statusRes) setStatus(statusRes || null);
      if (metricsRes) setMetrics(metricsRes.metrics || null);
      if (logRes) setTrainingLog(logRes.rows || []);
      if (ckptRes) setCheckpoints(ckptRes.checkpoints || []);
      if (samplesRes) {
        const nextSamples = samplesRes.samples || [];
        setSamples(nextSamples);
        setSelectedSample((current) =>
          nextSamples.length && !nextSamples.some((sample) => sample.name === current) ? nextSamples[nextSamples.length - 1].name : current,
        );
      }

      if (errors.length) {
        setRefreshError(`部分轻量数据暂不可用，已保留上一次成功数据：${errors.slice(0, 2).join("; ")}`);
      } else {
        markRefreshSuccess();
      }
    } finally {
      liveRefreshInFlight.current = false;
    }
  }, [markRefreshSuccess, selectedRunId]);

  const loadDatasetPreview = useCallback(async () => {
    if (!selectedRunId) {
      setDatasetPreview(null);
      return;
    }
    const runId = selectedRunId;
    try {
      const result = await api.datasetSamples(runId, {
        mode: datasetMode,
        limit: 4,
        include_tokens: true,
        token_limit: 80,
      });
      if (selectedRunIdRef.current !== runId) return;
      setDatasetPreview(result);
      markRefreshSuccess();
    } catch (err) {
      setDatasetPreview(null);
      console.warn("dataset preview failed", err);
    }
  }, [datasetMode, markRefreshSuccess, selectedRunId]);

  const loadSample = useCallback(async () => {
    if (!selectedRunId || !selectedSample) {
      setSampleText("");
      return;
    }
    const runId = selectedRunId;
    const sampleName = selectedSample;
    try {
      const result = await api.sample(runId, sampleName);
      if (selectedRunIdRef.current !== runId || selectedSampleRef.current !== sampleName) return;
      setSampleText(result.text || "");
      markRefreshSuccess();
    } catch (err) {
      markRefreshFailure("sample refresh failed", err);
    }
  }, [markRefreshFailure, markRefreshSuccess, selectedRunId, selectedSample]);

  useEffect(() => {
    loadRuns();
  }, [loadRuns]);

  useEffect(() => {
    loadRunStaticData();
    loadRunLiveData();
    loadRunDebugData();
  }, [loadRunStaticData, loadRunLiveData, loadRunDebugData]);

  useEffect(() => {
    loadDatasetPreview();
  }, [loadDatasetPreview]);

  useEffect(() => {
    loadSample();
  }, [loadSample]);

  useInterval(() => {
    loadRunLiveData();
  }, refreshIntervalMs, autoRefresh);

  useInterval(() => {
    loadRuns();
  }, 30000, autoRefresh);

  function handleApiBaseSubmit(event) {
    event.preventDefault();
    saveApiBaseUrlOverride(apiBaseInput);
    window.location.reload();
  }

  function handleApiBaseAuto() {
    clearApiBaseUrlOverride();
    window.location.reload();
  }

  function switchView(nextView) {
    setView(nextView);
    if (typeof window !== "undefined") {
      const url = new URL(window.location.href);
      if (nextView === "inspector") {
        url.searchParams.set("view", nextView);
        url.searchParams.set("tool", inspectorTool);
      } else {
        url.searchParams.delete("view");
        url.searchParams.delete("tool");
      }
      window.history.replaceState(null, "", url.toString());
    }
  }

  function switchInspectorTool(nextTool) {
    setInspectorTool(nextTool);
    if (typeof window !== "undefined" && view === "inspector") {
      const url = new URL(window.location.href);
      url.searchParams.set("view", "inspector");
      url.searchParams.set("tool", nextTool);
      window.history.replaceState(null, "", url.toString());
    }
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand-title">
          <div className="eyebrow">MLX GPT Lab</div>
          <h1>Training Dashboard</h1>
        </div>
        <div className="header-button-row">
          <div className="view-tabs">
            <button className={`toggle ${view === "dashboard" ? "on" : ""}`} onClick={() => switchView("dashboard")}>
              Dashboard
            </button>
            <button className={`toggle ${view === "inspector" ? "on" : ""}`} onClick={() => switchView("inspector")}>
              Model Inspector
            </button>
          </div>
          <button className={`toggle ${autoRefresh ? "on" : ""}`} onClick={() => setAutoRefresh((value) => !value)}>
            {autoRefresh ? <Pause size={15} /> : <Play size={15} />}
            Auto refresh {autoRefresh ? "on" : "off"}
          </button>
        </div>
      </header>

      <div className={`refresh-bar ${refreshError ? "warning" : "ok"}`}>
        <span className="refresh-message">
          {refreshError ? refreshError : `数据刷新正常 · 上次刷新 ${lastRefreshLabel}`}
        </span>
        <div className="header-meta refresh-controls">
          <form className="api-base-form" onSubmit={handleApiBaseSubmit}>
            <label htmlFor="api-base-url">API</label>
            <input
              id="api-base-url"
              value={apiBaseInput}
              onChange={(event) => setApiBaseInput(event.target.value)}
              placeholder="http://192.168.1.10:8765"
            />
            <span className="api-source">{API_BASE_URL_SOURCE}</span>
            <button type="submit" className="toggle">
              Save
            </button>
            <button type="button" className="toggle" onClick={handleApiBaseAuto}>
              Auto
            </button>
          </form>
          <label className="refresh-interval-control">
            Refresh
            <select
              value={refreshIntervalMs}
              onChange={(event) => setRefreshIntervalMs(Number(event.target.value))}
              disabled={!autoRefresh}
            >
              <option value={2000}>2s</option>
              <option value={4000}>4s</option>
              <option value={10000}>10s</option>
              <option value={30000}>30s</option>
            </select>
          </label>
        </div>
      </div>

      <section className="status-strip">
        <div>
          <Activity size={16} />
          <span>Health</span>
          <strong>{health?.ok ? "OK" : "暂无数据"}</strong>
        </div>
        <div>
          <span>Runs dir</span>
          <strong>{health?.runs_dir || "暂无数据"}</strong>
        </div>
        <div>
          <span>Current updated</span>
          <strong>{currentRun?.updated_at ? new Date(currentRun.updated_at).toLocaleString() : "暂无数据"}</strong>
        </div>
      </section>

      {view === "inspector" ? (
        <ModelInspector runs={runs} initialTool={inspectorTool} onToolChange={switchInspectorTool} />
      ) : (
        <div className="dashboard-grid">
          <RunSelector runs={runs} selectedRunId={selectedRunId} onSelect={setSelectedRunId} onRefresh={loadRuns} />
          <div className="main-column">
            <MetricsPanel status={status} metrics={metrics} checkpoints={checkpoints} />
            <LossChart rows={trainingLog} />
            <SampleViewer
              samples={samples}
              selectedSample={selectedSample}
              sampleText={sampleText}
              finalText={finalText}
              onSelectSample={setSelectedSample}
            />
            <DatasetPreview
              data={datasetPreview}
              mode={datasetMode}
              onModeChange={setDatasetMode}
              onRefresh={loadDatasetPreview}
            />
            <TokenizerResultPreview runId={selectedRunId} />
            <div className="two-column">
              <EvalViewer evalData={evalData} />
              <BenchmarkViewer benchmarkData={benchmarkData} />
            </div>
            <ConfigViewer config={config} />
          </div>
        </div>
      )}
    </main>
  );
}
