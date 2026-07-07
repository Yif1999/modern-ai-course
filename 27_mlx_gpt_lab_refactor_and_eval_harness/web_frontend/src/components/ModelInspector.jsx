import { useEffect, useState } from "react";
import AttentionInspector from "./AttentionInspector";
import ProbeLab from "./ProbeLab";
import TokenAtlas from "./TokenAtlas";

function normalizeTool(value) {
  return ["probe", "attention", "atlas"].includes(value) ? value : "probe";
}

export default function ModelInspector({ runs, initialTool = "probe", onToolChange }) {
  const [tool, setTool] = useState(() => normalizeTool(initialTool));

  useEffect(() => {
    setTool(normalizeTool(initialTool));
  }, [initialTool]);

  function selectTool(nextTool) {
    const normalized = normalizeTool(nextTool);
    setTool(normalized);
    onToolChange?.(normalized);
  }

  return (
    <div className="inspector-shell">
      <section className="panel inspector-hero">
        <div>
          <h2>Model Inspector</h2>
          <div className="panel-note">
            一个子页面集中查看 checkpoint 的 next-token 分布、attention head 行为和 token 向量空间。
          </div>
        </div>
        <div className="inspector-tabs">
          <button className={`toggle ${tool === "probe" ? "on" : ""}`} onClick={() => selectTool("probe")}>
            Next Token / Trace
          </button>
          <button className={`toggle ${tool === "attention" ? "on" : ""}`} onClick={() => selectTool("attention")}>
            Attention Heads
          </button>
          <button className={`toggle ${tool === "atlas" ? "on" : ""}`} onClick={() => selectTool("atlas")}>
            Token Atlas
          </button>
        </div>
      </section>

      {tool === "attention" ? <AttentionInspector runs={runs} /> : null}
      {tool === "atlas" ? <TokenAtlas runs={runs} /> : null}
      {tool === "probe" ? <ProbeLab runs={runs} /> : null}
    </div>
  );
}
