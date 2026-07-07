import { useEffect, useMemo, useRef, useState } from "react";

function fmt(value, digits = 3) {
  const number = Number(value);
  if (value === null || value === undefined || !Number.isFinite(number)) return "暂无数据";
  return number.toFixed(digits);
}

function scale(value, min, max, start, end) {
  if (max === min) return (start + end) / 2;
  return start + ((value - min) / (max - min)) * (end - start);
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function linePath(points) {
  if (points.length === 0) return "";
  return points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(" ");
}

function seriesPoints(rows, key, xMin, xMax, yMin, yMax, plot) {
  return rows
    .filter((row) => row.step !== undefined && row[key] !== undefined && row[key] !== null)
    .map((row) => ({
      x: scale(Number(row.step), xMin, xMax, plot.left, plot.right),
      y: scale(Number(row[key]), yMin, yMax, plot.bottom, plot.top),
      step: Number(row.step),
      value: Number(row[key]),
      row,
    }));
}

function rangeLabel(start, end) {
  if (!Number.isFinite(start) || !Number.isFinite(end)) return "暂无数据";
  return `${Math.round(start).toLocaleString()} - ${Math.round(end).toLocaleString()}`;
}

export default function LossChart({ rows }) {
  const [hover, setHover] = useState(null);
  const [viewRange, setViewRange] = useState(null);
  const [rangeMode, setRangeMode] = useState("full");
  const [dragHandle, setDragHandle] = useState(null);
  const chartRef = useRef(null);
  const rangeRef = useRef(null);
  const [svgWidth, setSvgWidth] = useState(820);

  useEffect(() => {
    const element = chartRef.current;
    if (!element) return undefined;

    const updateWidth = () => {
      const rect = element.getBoundingClientRect();
      if (!rect.width) return;
      const nextWidth = Math.max(620, Math.round((rect.width / 300) * 270));
      setSvgWidth((current) => (Math.abs(current - nextWidth) > 2 ? nextWidth : current));
    };

    updateWidth();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", updateWidth);
      return () => window.removeEventListener("resize", updateWidth);
    }

    const observer = new ResizeObserver(updateWidth);
    observer.observe(element);
    return () => observer.disconnect();
  }, [rows?.length]);

  const prepared = useMemo(() => {
    const cleanRows = (rows || []).filter((row) => row.step !== undefined && (row.train_loss !== undefined || row.val_loss !== undefined));
    const source = cleanRows
      .map((row) => ({ ...row, step: Number(row.step) }))
      .filter((row) => Number.isFinite(row.step))
      .sort((a, b) => a.step - b.step);

    const steps = source.map((row) => row.step);
    const fullValues = source
      .flatMap((row) => [row.train_loss, row.val_loss])
      .filter((value) => value !== undefined && value !== null)
      .map(Number)
      .filter(Number.isFinite);

    if (source.length === 0 || steps.length === 0 || fullValues.length === 0) {
      return { rows: source, empty: true };
    }

    const fullXMin = Math.min(...steps);
    const fullXMax = Math.max(...steps);
    const minWindow = Math.max(1, (fullXMax - fullXMin) * 0.002);
    const recentCount = rangeMode.startsWith("recent-") ? Number(rangeMode.replace("recent-", "")) : null;
    const recentStart = Number.isFinite(recentCount) ? steps[Math.max(0, steps.length - recentCount)] : null;
    const requestedStart = rangeMode === "full" ? fullXMin : recentStart ?? viewRange?.start ?? fullXMin;
    const requestedEnd = rangeMode === "full" || Number.isFinite(recentCount) ? fullXMax : viewRange?.end ?? fullXMax;
    let rangeStart = clamp(Math.min(requestedStart, requestedEnd), fullXMin, fullXMax);
    let rangeEnd = clamp(Math.max(requestedStart, requestedEnd), fullXMin, fullXMax);
    if (rangeEnd - rangeStart < minWindow) {
      rangeEnd = clamp(rangeStart + minWindow, fullXMin, fullXMax);
      rangeStart = clamp(rangeEnd - minWindow, fullXMin, fullXMax);
    }

    const visibleSource = source.filter((row) => row.step >= rangeStart && row.step <= rangeEnd);
    const effectiveSource = visibleSource.length ? visibleSource : source;
    const evalRows = effectiveSource.filter((row) => row.val_loss !== undefined && row.val_loss !== null);
    const heartbeatRows = effectiveSource.filter((row) => row.val_loss === undefined || row.val_loss === null);

    const values = effectiveSource
      .flatMap((row) => [row.train_loss, row.val_loss])
      .filter((value) => value !== undefined && value !== null)
      .map(Number)
      .filter(Number.isFinite);
    const xMin = Math.min(...effectiveSource.map((row) => row.step));
    const xMax = Math.max(...effectiveSource.map((row) => row.step));
    const rawYMin = Math.min(...values);
    const rawYMax = Math.max(...values);
    const yRange = rawYMax - rawYMin;
    const yPad = yRange > 0 ? yRange * 0.08 : Math.max(Math.abs(rawYMax) * 0.01, 0.01);
    const yMin = rawYMin - yPad;
    const yMax = rawYMax + yPad;
    const plot = { left: 48, right: Math.max(140, svgWidth - 8), top: 18, bottom: 238 };

    const train = seriesPoints(heartbeatRows, "train_loss", xMin, xMax, yMin, yMax, plot);
    const val = seriesPoints(effectiveSource, "val_loss", xMin, xMax, yMin, yMax, plot);
    const hoverPoints = [...train.map((point) => ({ ...point, series: "train" })), ...val.map((point) => ({ ...point, series: "val" }))].sort(
      (a, b) => a.x - b.x,
    );
    const fullEvalRows = source.filter((row) => row.val_loss !== undefined && row.val_loss !== null);
    const fullHeartbeatRows = source.filter((row) => row.val_loss === undefined || row.val_loss === null);
    const bestVals = fullEvalRows.map((row) => Number(row.val_loss)).filter(Number.isFinite);
    const rangeStartPercent = fullXMax === fullXMin ? 0 : ((rangeStart - fullXMin) / (fullXMax - fullXMin)) * 100;
    const rangeEndPercent = fullXMax === fullXMin ? 100 : ((rangeEnd - fullXMin) / (fullXMax - fullXMin)) * 100;

    return {
      rows: effectiveSource,
      empty: false,
      xMin,
      xMax,
      fullXMin,
      fullXMax,
      rangeStart,
      rangeEnd,
      rangeStartPercent,
      rangeEndPercent,
      rangeMode,
      isFullRange: rangeMode === "full" || (Math.abs(rangeStart - fullXMin) < 1 && Math.abs(rangeEnd - fullXMax) < 1),
      yMin,
      yMax,
      svgWidth,
      plot,
      trainPath: linePath(train),
      valPath: linePath(val),
      hoverPoints,
      latest: source[source.length - 1],
      latestTrain: [...fullHeartbeatRows].reverse().find((row) => Number.isFinite(Number(row.train_loss)))?.train_loss,
      latestVal: [...fullEvalRows].reverse().find((row) => Number.isFinite(Number(row.val_loss)))?.val_loss,
      latestValStep: [...fullEvalRows].reverse().find((row) => Number.isFinite(Number(row.val_loss)))?.step,
      bestVal: bestVals.length ? Math.min(...bestVals) : null,
      totalRows: cleanRows.length,
      visibleRows: effectiveSource.length,
      evalRows: evalRows.length,
      heartbeatRows: heartbeatRows.length,
    };
  }, [rangeMode, rows, svgWidth, viewRange]);

  useEffect(() => {
    if (prepared.empty || viewRange === null || rangeMode !== "custom") return;
    if (viewRange.start < prepared.fullXMin || viewRange.end > prepared.fullXMax || viewRange.end <= viewRange.start) {
      setViewRange({
        start: clamp(viewRange.start, prepared.fullXMin, prepared.fullXMax),
        end: clamp(viewRange.end, prepared.fullXMin, prepared.fullXMax),
      });
    }
  }, [prepared.empty, prepared.fullXMax, prepared.fullXMin, rangeMode, viewRange]);

  function stepFromClientX(clientX) {
    const rect = rangeRef.current?.getBoundingClientRect();
    if (!rect || prepared.empty || prepared.fullXMax === prepared.fullXMin) return prepared.fullXMin;
    const ratio = clamp((clientX - rect.left) / rect.width, 0, 1);
    return prepared.fullXMin + ratio * (prepared.fullXMax - prepared.fullXMin);
  }

  function updateRangeFromPointer(handle, clientX) {
    if (prepared.empty) return;
    const value = stepFromClientX(clientX);
    const minWindow = Math.max(1, (prepared.fullXMax - prepared.fullXMin) * 0.002);
    const currentStart = prepared.rangeStart;
    const currentEnd = prepared.rangeEnd;
    if (handle === "start") {
      setRangeMode("custom");
      setViewRange({ start: clamp(value, prepared.fullXMin, currentEnd - minWindow), end: currentEnd });
    } else {
      setRangeMode("custom");
      setViewRange({ start: currentStart, end: clamp(value, currentStart + minWindow, prepared.fullXMax) });
    }
  }

  useEffect(() => {
    if (!dragHandle) return undefined;
    const onMove = (event) => updateRangeFromPointer(dragHandle, event.clientX);
    const onUp = () => setDragHandle(null);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp, { once: true });
    window.addEventListener("pointercancel", onUp, { once: true });
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };
  }, [dragHandle, prepared]);

  function handleRangeRailPointerDown(event) {
    if (prepared.empty) return;
    const targetStep = stepFromClientX(event.clientX);
    const distanceToStart = Math.abs(targetStep - prepared.rangeStart);
    const distanceToEnd = Math.abs(targetStep - prepared.rangeEnd);
    const handle = distanceToStart <= distanceToEnd ? "start" : "end";
    setDragHandle(handle);
    updateRangeFromPointer(handle, event.clientX);
  }

  function setRecentWindow(count) {
    setRangeMode(`recent-${count}`);
    setViewRange(null);
    setHover(null);
  }

  function resetRange() {
    setRangeMode("full");
    setViewRange(null);
    setHover(null);
  }

  function handlePointerMove(event) {
    if (prepared.empty || !prepared.hoverPoints?.length) return;
    let x;
    let y;
    const svg = event.currentTarget;
    const matrix = svg.getScreenCTM?.();
    if (matrix && svg.createSVGPoint) {
      const point = svg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      const svgPoint = point.matrixTransform(matrix.inverse());
      x = svgPoint.x;
      y = svgPoint.y;
    } else {
      const rect = svg.getBoundingClientRect();
      x = ((event.clientX - rect.left) / rect.width) * prepared.svgWidth;
      y = ((event.clientY - rect.top) / rect.height) * 270;
    }
    if (x < prepared.plot.left || x > prepared.plot.right || y < prepared.plot.top - 10 || y > prepared.plot.bottom + 10) {
      setHover(null);
      return;
    }

    let nearest = prepared.hoverPoints[0];
    let bestDistance = Infinity;
    for (const point of prepared.hoverPoints) {
      const distance = Math.abs(point.x - x) + Math.abs(point.y - y) * 0.2;
      if (distance < bestDistance) {
        nearest = point;
        bestDistance = distance;
      }
    }
    setHover(nearest);
  }

  const tooltip = hover
    ? {
        x: hover.x > prepared.svgWidth - 200 ? hover.x - 184 : hover.x + 12,
        y: hover.y < 76 ? hover.y + 16 : hover.y - 72,
      }
    : null;

  return (
    <section className="panel chart-panel">
      <div className="panel-header">
        <div>
          <h2>Loss Chart</h2>
          <div className="panel-note">
            visible {prepared.visibleRows || 0} / {prepared.totalRows || 0} rows · range{" "}
            {prepared.empty ? "暂无数据" : rangeLabel(prepared.rangeStart, prepared.rangeEnd)}
          </div>
        </div>
        {!prepared.empty && (
          <div className="chart-stats">
            <span>step {prepared.latest?.step ?? "暂无数据"}</span>
            <span>train {fmt(prepared.latestTrain)}</span>
            <span>
              val {fmt(prepared.latestVal)}
              {prepared.latestValStep ? ` @ ${prepared.latestValStep}` : ""}
            </span>
            <span>best val {fmt(prepared.bestVal)}</span>
          </div>
        )}
      </div>
      {prepared.empty ? (
        <div className="empty">暂无数据</div>
      ) : (
        <div className="lightweight-chart" ref={chartRef}>
          <svg
            viewBox={`0 0 ${prepared.svgWidth} 270`}
            role="img"
            aria-label="training and validation loss chart"
            onMouseMove={handlePointerMove}
            onMouseLeave={() => setHover(null)}
          >
            <line x1={prepared.plot.left} y1={prepared.plot.top} x2={prepared.plot.left} y2={prepared.plot.bottom} />
            <line x1={prepared.plot.left} y1={prepared.plot.bottom} x2={prepared.plot.right} y2={prepared.plot.bottom} />
            {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
              const y = scale(tick, 0, 1, prepared.plot.bottom, prepared.plot.top);
              const value = prepared.yMin + (prepared.yMax - prepared.yMin) * tick;
              return (
                <g key={tick}>
                  <line className="grid-line" x1={prepared.plot.left} y1={y} x2={prepared.plot.right} y2={y} />
                  <text x={prepared.plot.left - 8} y={y + 4} textAnchor="end">
                    {fmt(value, 2)}
                  </text>
                </g>
              );
            })}
            <text x={prepared.plot.left} y={prepared.plot.bottom + 20}>
              step {Math.round(prepared.xMin).toLocaleString()}
            </text>
            <text x={prepared.plot.right} y={prepared.plot.bottom + 20} textAnchor="end">
              step {Math.round(prepared.xMax).toLocaleString()}
            </text>
            <path className="loss-line train" d={prepared.trainPath} />
            <path className="loss-line val" d={prepared.valPath} />
            <rect
              className="chart-hover-capture"
              x={prepared.plot.left}
              y={prepared.plot.top}
              width={prepared.plot.right - prepared.plot.left}
              height={prepared.plot.bottom - prepared.plot.top}
            />
            {hover && tooltip ? (
              <g className="chart-hover-layer">
                <line className="chart-hover-line" x1={hover.x} y1={prepared.plot.top} x2={hover.x} y2={prepared.plot.bottom} />
                <circle className={`chart-hover-dot ${hover.series}`} cx={hover.x} cy={hover.y} r="4.2" />
                <g transform={`translate(${tooltip.x}, ${tooltip.y})`}>
                  <rect className="chart-tooltip-bg" width="172" height="62" rx="7" />
                  <text className="chart-tooltip-title" x="10" y="18">
                    step {hover.step.toLocaleString()}
                  </text>
                  <text className="chart-tooltip-text" x="10" y="37">
                    {hover.series} loss {fmt(hover.value, 4)}
                  </text>
                  <text className="chart-tooltip-text muted" x="10" y="53">
                    {hover.series === "train" ? "heartbeat / train" : "eval / validation"}
                  </text>
                </g>
              </g>
            ) : null}
          </svg>

          <div className="chart-range-toolbar">
            <div className="chart-legend">
              <span className="legend-item train">train loss</span>
              <span className="legend-item val">val loss</span>
            </div>
            <div className="chart-range-actions">
              <button type="button" className={rangeMode === "full" ? "mini-button active" : "mini-button"} onClick={resetRange}>
                全程
              </button>
              <button type="button" className={rangeMode === "recent-500" ? "mini-button active" : "mini-button"} onClick={() => setRecentWindow(500)}>
                最近 500
              </button>
              <button type="button" className={rangeMode === "recent-100" ? "mini-button active" : "mini-button"} onClick={() => setRecentWindow(100)}>
                最近 100
              </button>
            </div>
          </div>

          <div className="chart-range-control" ref={rangeRef} onPointerDown={handleRangeRailPointerDown}>
            <div className="chart-range-track" />
            <div
              className="chart-range-selection"
              style={{
                left: `${prepared.rangeStartPercent}%`,
                width: `${Math.max(0.6, prepared.rangeEndPercent - prepared.rangeStartPercent)}%`,
              }}
            />
            <button
              type="button"
              className="chart-range-handle start"
              style={{ left: `${prepared.rangeStartPercent}%` }}
              aria-label="调整 loss 曲线起点"
              onPointerDown={(event) => {
                event.stopPropagation();
                setDragHandle("start");
                updateRangeFromPointer("start", event.clientX);
              }}
            />
            <button
              type="button"
              className="chart-range-handle end"
              style={{ left: `${prepared.rangeEndPercent}%` }}
              aria-label="调整 loss 曲线终点"
              onPointerDown={(event) => {
                event.stopPropagation();
                setDragHandle("end");
                updateRangeFromPointer("end", event.clientX);
              }}
            />
          </div>
          <div className="chart-range-labels">
            <span>全程 step {Math.round(prepared.fullXMin).toLocaleString()}</span>
            <strong>当前窗口 step {rangeLabel(prepared.rangeStart, prepared.rangeEnd)}</strong>
            <span>step {Math.round(prepared.fullXMax).toLocaleString()}</span>
          </div>
        </div>
      )}
    </section>
  );
}
