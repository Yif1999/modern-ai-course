import { useEffect, useMemo, useState } from "react";

const PAGE_SIZE = 8;

function sampleSortKey(sample) {
  const stamp = sample?.updated_at ? Date.parse(sample.updated_at) : NaN;
  if (Number.isFinite(stamp)) return stamp;
  const match = String(sample?.name || "").match(/(\d+)/g);
  return match ? Number(match[match.length - 1]) : 0;
}

export default function SampleViewer({ samples, selectedSample, sampleText, finalText, onSelectSample }) {
  const [page, setPage] = useState(1);

  const sortedSamples = useMemo(
    () => [...(samples || [])].sort((a, b) => sampleSortKey(b) - sampleSortKey(a) || String(b.name).localeCompare(String(a.name))),
    [samples],
  );

  const totalPages = Math.max(1, Math.ceil(sortedSamples.length / PAGE_SIZE));
  const clampedPage = Math.min(page, totalPages);
  const pageStart = (clampedPage - 1) * PAGE_SIZE;
  const pageSamples = sortedSamples.slice(pageStart, pageStart + PAGE_SIZE);

  useEffect(() => {
    if (page !== clampedPage) setPage(clampedPage);
  }, [clampedPage, page]);

  useEffect(() => {
    if (!selectedSample || sortedSamples.length === 0) return;
    const index = sortedSamples.findIndex((sample) => sample.name === selectedSample);
    if (index >= 0) setPage(Math.floor(index / PAGE_SIZE) + 1);
  }, [selectedSample, sortedSamples]);

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Samples</h2>
        <span className="panel-note">
          latest first · {sortedSamples.length} files
        </span>
      </div>
      <div className="sample-layout">
        <div className="sample-list">
          {sortedSamples.length === 0 ? (
            <div className="empty">暂无数据</div>
          ) : (
            <>
              <div className="sample-list-pager">
                <button onClick={() => setPage((value) => Math.max(1, value - 1))} disabled={clampedPage <= 1}>
                  Prev
                </button>
                <span>
                  {clampedPage} / {totalPages}
                </span>
                <button onClick={() => setPage((value) => Math.min(totalPages, value + 1))} disabled={clampedPage >= totalPages}>
                  Next
                </button>
              </div>
              {pageSamples.map((sample) => (
                <button
                  key={sample.name}
                  className={`sample-button ${selectedSample === sample.name ? "active" : ""}`}
                  onClick={() => onSelectSample(sample.name)}
                >
                  <span>{sample.name}</span>
                  {sample.updated_at && <small>{new Date(sample.updated_at).toLocaleString()}</small>}
                </button>
              ))}
            </>
          )}
        </div>
        <div className="sample-content">
          <h3>Selected Sample</h3>
          <pre>{sampleText || "暂无数据"}</pre>
          <h3>Final Generated Text</h3>
          <pre>{finalText || "暂无数据"}</pre>
        </div>
      </div>
    </section>
  );
}
