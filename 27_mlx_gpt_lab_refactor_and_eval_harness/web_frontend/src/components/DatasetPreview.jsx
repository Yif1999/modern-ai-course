import { useMemo } from "react";
import { RefreshCw } from "lucide-react";
import { datasetSampleKey } from "./datasetSampleKey";

function fmt(value, digits = 2) {
  if (value === null || value === undefined || value === "") return "暂无数据";
  if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(digits);
  return String(value);
}

function percent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "暂无数据";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function basename(path) {
  if (!path) return "暂无数据";
  return String(path).split("/").filter(Boolean).slice(-2).join("/");
}

function SampleText({ text }) {
  const value = text || "暂无文本";
  return (
    <div className="sample-text-preview">
      <pre>{value}</pre>
    </div>
  );
}

export default function DatasetPreview({ data, mode, onModeChange, onRefresh }) {
  const meta = data?.metadata || {};
  const samples = data?.samples || [];
  const categoryRows = meta.category_summary || [];
  const visibleCategoryRows = categoryRows.slice(0, 6);
  const visibleCategoryShare = visibleCategoryRows.reduce(
    (total, row) => total + Number(meta.category_share?.[row.category] || 0),
    0,
  );
  const visibleCategoryDocs = visibleCategoryRows.reduce((total, row) => total + Number(row.docs || 0), 0);
  const totalCategoryCount = Object.keys(meta.category_share || {}).length || categoryRows.length;
  const hiddenCategoryCount = Math.max(0, totalCategoryCount - visibleCategoryRows.length);
  const otherCategoryShare = Math.max(0, 1 - visibleCategoryShare);
  const otherCategoryDocs =
    hiddenCategoryCount > 0 && meta.document_count
      ? Math.max(0, Number(meta.document_count) - visibleCategoryDocs)
      : 0;
  const tokenization = data?.tokenization || {};
  const sampleEntries = useMemo(
    () => samples.map((sample, index) => ({ key: datasetSampleKey(sample, index), sample, index })),
    [samples],
  );

  return (
    <section className="panel dataset-preview">
      <div className="panel-header">
        <div>
          <h2>Dataset Preview</h2>
          <div className="panel-note">原始训练样本预览</div>
        </div>
        <div className="panel-actions">
          <select value={mode} onChange={(event) => onModeChange(event.target.value)}>
            <option value="random">random samples</option>
            <option value="head">first samples</option>
          </select>
          <button className="icon-button" onClick={onRefresh} title="刷新数据样本">
            <RefreshCw size={16} />
          </button>
        </div>
      </div>

      <div className="dataset-summary compact">
        <div className="metric">
          <span>source</span>
          <strong>{data?.source || "暂无数据"}</strong>
        </div>
        <div className="metric">
          <span>docs</span>
          <strong>{fmt(meta.document_count, 0)}</strong>
        </div>
        <div className="metric">
          <span>raw chars</span>
          <strong>{fmt(meta.raw_chars, 0)}</strong>
        </div>
        <div className="metric">
          <span>total tokens</span>
          <strong>{fmt(meta.total_tokens, 0)}</strong>
        </div>
        <div className="metric">
          <span>vocab</span>
          <strong>{fmt(meta.vocab_size, 0)}</strong>
        </div>
        <div className="metric">
          <span>chars/token</span>
          <strong>{fmt(meta.chars_per_token, 3)}</strong>
        </div>
      </div>

      <details className="dataset-paths">
        <summary>数据文件路径</summary>
        <div>
          <span>metadata: {basename(meta.metadata_path)}</span>
          <span>docs: {basename(meta.docs_jsonl_path)}</span>
          <span>tokenizer: {basename(tokenization.tokenizer_path)}</span>
        </div>
      </details>

      {data?.note && <div className="dataset-note">{data.note}</div>}
      {tokenization.tokenizer_error && <div className="dataset-note warning">{tokenization.tokenizer_error}</div>}

      {categoryRows.length > 0 && (
        <>
          <div className="category-strip-title">
            <span>category mix</span>
            <small>top 6 + other</small>
          </div>
          <div className="category-strip">
            {visibleCategoryRows.map((row) => (
              <div className="category-pill" key={row.category}>
                <strong>{row.category}</strong>
                <span>{percent(meta.category_share?.[row.category])}</span>
                <small>{fmt(row.docs, 0)} docs</small>
              </div>
            ))}
            {hiddenCategoryCount > 0 && (
              <div className="category-pill muted" key="__other__">
                <strong>other</strong>
                <span>{percent(otherCategoryShare)}</span>
                <small>
                  {hiddenCategoryCount} categories · {fmt(otherCategoryDocs, 0)} docs
                </small>
              </div>
            )}
          </div>
        </>
      )}

      <div className="dataset-sample-list">
        {samples.length === 0 ? (
          <div className="empty">暂无数据</div>
        ) : (
          sampleEntries.map((entry) => {
            const sample = entry.sample;
            return (
            <article className="dataset-sample-card" key={entry.key}>
              <div className="dataset-sample-meta">
                <span className="badge ok">{sample.category || "unknown"}</span>
                <span className="badge file-ok">{sample.source_type || "unknown"}</span>
                <span>{sample.source_name || "unknown"}</span>
                <span>{fmt(sample.char_count, 0)} chars</span>
                {sample.turn_count !== null && sample.turn_count !== undefined ? <span>{sample.turn_count} turns</span> : null}
                {sample.tokenization?.token_count !== undefined ? <span>{fmt(sample.tokenization.token_count, 0)} tokens</span> : null}
              </div>
              <SampleText text={sample.text} />
            </article>
            );
          })
        )}
      </div>
    </section>
  );
}
