# MLX GPT Lab Dashboard Backend

这是只读 FastAPI 后端，只读取：

```text
../outputs/runs/
```

也可以用环境变量覆盖：

```bash
export MLX_GPT_LAB_RUNS_DIR=/absolute/path/to/runs
```

## 安装

```bash
cd /Volumes/T7/Dev/ai-lab/27_mlx_gpt_lab_refactor_and_eval_harness/web_backend
python -m pip install -r requirements.txt
```

## 启动

```bash
uvicorn app:app --host 0.0.0.0 --port 8765
```

`--host 0.0.0.0` 表示允许局域网内其他机器访问。可以用下面命令查看本机局域网 IP：

```bash
ipconfig getifaddr en0
```

如果返回 `192.168.1.10`，第二台机器可以访问：

```text
http://192.168.1.10:8765/api/health
```

同时前端需要用 `npm run dev` 启动，因为当前 Vite 配置也是 `--host 0.0.0.0`。

## 验证

```text
http://localhost:8765/api/health
http://localhost:8765/api/runs
```

## API

- `GET /api/health`
- `GET /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/config`
- `GET /api/runs/{run_id}/training-log`
- `GET /api/runs/{run_id}/metrics`
- `GET /api/runs/{run_id}/status`
- `GET /api/runs/{run_id}/samples`
- `GET /api/runs/{run_id}/samples/{sample_name}`
- `GET /api/runs/{run_id}/dataset-samples`
- `GET /api/runs/{run_id}/final-text`
- `GET /api/runs/{run_id}/eval-results`
- `GET /api/runs/{run_id}/benchmark`
- `GET /api/runs/{run_id}/checkpoints`

## 只读约束

本后端不会：

- 启动训练
- 停止训练
- 删除 run
- 删除 checkpoint
- 执行 shell 命令
- 上传数据

## 训练输出建议

后续训练脚本建议持续输出：

```text
training_log.jsonl
metrics.json
status.json
samples/
final_generated_text.txt
eval_results.jsonl
benchmark_results.json
```

如果希望 Dashboard 预览训练数据样本，建议 `config.json` 里记录：

```json
{
  "metadata_path": "/absolute/or/project/relative/path/to/metadata.json"
}
```

对应 `metadata.json` 建议包含：

```json
{
  "docs_jsonl_path": "/path/to/raw_docs.jsonl",
  "corpus_path": "/path/to/corpus.txt",
  "tokenizer_path": "/path/to/tokenizer.json",
  "category_summary": {},
  "category_share": {}
}
```

`dataset-samples` API 会优先读取 `docs_jsonl_path`，并用 `tokenizer_path` 生成样本分词预览；如果没有 JSONL，则退回到 `corpus_path` 的文本切片。

特别建议训练过程中持续更新 `status.json`：

```json
{
  "run_id": "qwen_dense_tiny_debug_20260612_120000",
  "step": 1200,
  "train_loss": 4.2,
  "val_loss": 4.5,
  "tokens_seen": 3932160,
  "tokens_per_second": 1200.5,
  "updated_at": "2026-06-12T12:00:00"
}
```

## 后续可扩展

- token-level top-k viewer
- logits distribution viewer
- attention heatmap viewer
- prompt playground
- checkpoint comparison
- multi-run comparison
- NAS 部署
