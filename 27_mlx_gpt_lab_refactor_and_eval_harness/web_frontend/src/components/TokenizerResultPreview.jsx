import { useCallback, useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, RefreshCw } from "lucide-react";
import { api } from "../api";

function fmt(value, digits = 2) {
  if (value === null || value === undefined || value === "") return "暂无数据";
  if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(digits);
  return String(value);
}

export default function TokenizerResultPreview({ runId }) {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(120);
  const [query, setQuery] = useState("");
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [selectedTokenId, setSelectedTokenId] = useState(null);

  const loadVocab = useCallback(async () => {
    if (!runId) {
      setData(null);
      return;
    }
    try {
      const result = await api.tokenizerVocab(runId, { page, page_size: pageSize, query });
      setData(result);
      setError("");
    } catch (err) {
      setData(null);
      setError(err.message || String(err));
    }
  }, [runId, page, pageSize, query]);

  useEffect(() => {
    loadVocab();
  }, [loadVocab]);

  useEffect(() => {
    setPage(1);
  }, [runId, query, pageSize]);

  const totalPages = data?.total_pages || 0;
  const rows = data?.rows || [];
  const selectedToken = rows.find((row) => row.id === selectedTokenId) || rows[0] || null;

  useEffect(() => {
    if (rows.length === 0) {
      setSelectedTokenId(null);
      return;
    }
    if (!rows.some((row) => row.id === selectedTokenId)) {
      setSelectedTokenId(rows[0].id);
    }
  }, [rows, selectedTokenId]);

  return (
    <section className="panel tokenizer-result-preview">
      <div className="panel-header">
        <div>
          <h2>Tokenizer Result Preview</h2>
          <div className="panel-note">完整 tokenizer 词表分页预览</div>
        </div>
        <div className="panel-actions">
          <input
            className="tokenizer-search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="搜索 decoded 或 token id"
          />
          <select value={pageSize} onChange={(event) => setPageSize(Number(event.target.value))}>
            <option value={120}>120 / page</option>
            <option value={240}>240 / page</option>
            <option value={480}>480 / page</option>
          </select>
          <button className="icon-button" onClick={loadVocab} title="刷新 tokenizer 词表">
            <RefreshCw size={16} />
          </button>
        </div>
      </div>

      {error && <div className="dataset-note warning">{error}</div>}

      <div className="tokenizer-compact-meta">
        <strong>{data?.tokenizer_name || "暂无 tokenizer"}</strong>
        <span>vocab {fmt(data?.vocab_size, 0)}</span>
        <span>matched {fmt(data?.total, 0)}</span>
        <span>
          page {fmt(data?.page, 0)} / {fmt(totalPages, 0)}
        </span>
      </div>

      <div className="tokenizer-pager">
        <button className="sample-token-button" disabled={!data?.ok || page <= 1} onClick={() => setPage((value) => Math.max(1, value - 1))}>
          <ChevronLeft size={14} />
          Prev
        </button>
        <span>
          showing {rows.length.toLocaleString()} tokens
          {query ? ` · query: ${query}` : ""}
        </span>
        <button
          className="sample-token-button"
          disabled={!data?.ok || (totalPages && page >= totalPages)}
          onClick={() => setPage((value) => value + 1)}
        >
          Next
          <ChevronRight size={14} />
        </button>
      </div>

      {selectedToken && (
        <div className="selected-token-strip">
          <span>Selected</span>
          <strong>#{selectedToken.id}</strong>
          <code>{selectedToken.display || "∅"}</code>
          <small>{selectedToken.kind || "text"}</small>
        </div>
      )}

      {data?.ok === false ? (
        <div className="empty">{data.error || "tokenizer 不可用"}</div>
      ) : rows.length === 0 ? (
        <div className="empty">暂无数据</div>
      ) : (
        <div className="token-vocab-grid">
          {rows.map((row) => (
            <button
              type="button"
              className={`token-vocab-cell ${row.is_special ? "special" : ""} ${row.readable === false ? "unreadable" : ""} ${
                selectedToken?.id === row.id ? "selected" : ""
              }`}
              key={row.id}
              title={`${row.id}: ${row.display || "∅"}`}
              onClick={() => setSelectedTokenId(row.id)}
            >
              <span className="token-vocab-id">{row.id}</span>
              <strong>{row.display || "∅"}</strong>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
