import { useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, RefreshCw } from "lucide-react";

function formatDate(value) {
  if (!value) return "暂无数据";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function stateLabel(state) {
  return {
    running: "running",
    completed: "completed",
    failed: "failed",
    stopped: "stopped",
    stale: "stale",
    initializing: "initializing",
    unknown: "unknown",
  }[state || "unknown"];
}

export default function RunSelector({ runs, selectedRunId, onSelect, onRefresh }) {
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(8);
  const totalPages = Math.max(1, Math.ceil(runs.length / pageSize));
  const visibleRuns = useMemo(
    () => runs.slice(page * pageSize, page * pageSize + pageSize),
    [page, pageSize, runs],
  );

  useEffect(() => {
    setPage((current) => Math.min(current, totalPages - 1));
  }, [totalPages]);

  useEffect(() => {
    if (!selectedRunId) return;
    const index = runs.findIndex((run) => run.run_id === selectedRunId);
    if (index >= 0) {
      setPage(Math.floor(index / pageSize));
    }
  }, [pageSize, runs, selectedRunId]);

  function handlePageSizeChange(event) {
    setPageSize(Number(event.target.value));
    setPage(0);
  }

  return (
    <section className="panel run-selector">
      <div className="panel-header">
        <h2>Runs</h2>
        <button className="icon-button" onClick={onRefresh} title="刷新 run 列表">
          <RefreshCw size={16} />
        </button>
      </div>
      <div className="run-list-toolbar">
        <span>{runs.length} runs</span>
        <label>
          per page
          <select value={pageSize} onChange={handlePageSizeChange}>
            <option value={8}>8</option>
            <option value={12}>12</option>
            <option value={20}>20</option>
          </select>
        </label>
      </div>
      <div className="run-list">
        {runs.length === 0 ? (
          <div className="empty">暂无数据</div>
        ) : (
          visibleRuns.map((run) => (
            <button
              key={run.run_id}
              className={`run-item ${selectedRunId === run.run_id ? "active" : ""}`}
              onClick={() => onSelect(run.run_id)}
            >
              <span className="run-id">{run.run_id}</span>
              <span className="run-time">
                {formatDate(run.updated_at)}
                {run.latest_step !== null && run.latest_step !== undefined ? ` · step ${run.latest_step}` : ""}
              </span>
              <span className="run-badges">
                <span className={`badge state ${run.state || "unknown"}`}>{stateLabel(run.state)}</span>
                <span className={run.has_config ? "badge ok" : "badge"}>config</span>
                <span className={run.has_training_log ? "badge ok" : "badge"}>log</span>
                <span className={run.has_samples ? "badge ok" : "badge"}>samples</span>
                <span className={run.has_final_text ? "badge file-ok" : "badge"}>final file</span>
              </span>
            </button>
          ))
        )}
      </div>
      {runs.length > 0 ? (
        <div className="run-pager">
          <button
            className="icon-button"
            onClick={() => setPage((current) => Math.max(0, current - 1))}
            disabled={page === 0}
            title="上一页"
          >
            <ChevronLeft size={16} />
          </button>
          <span>
            page {page + 1} / {totalPages}
          </span>
          <button
            className="icon-button"
            onClick={() => setPage((current) => Math.min(totalPages - 1, current + 1))}
            disabled={page >= totalPages - 1}
            title="下一页"
          >
            <ChevronRight size={16} />
          </button>
        </div>
      ) : null}
    </section>
  );
}
