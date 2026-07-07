from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from checkpoint_probe import (
    attention_probe,
    checkpoint_compare_probe,
    generation_trace_probe,
    next_token_probe,
    token_neighborhood_probe,
    tokenize_prompt,
    unload_probe,
)
from run_reader import (
    ensure_demo_run,
    get_benchmark,
    get_config,
    get_dataset_samples,
    get_eval_results,
    get_final_text,
    get_metrics,
    get_runs_dir,
    get_sample_text,
    get_status,
    get_system_resources,
    get_tokenizer_vocab,
    list_checkpoints,
    list_runs,
    list_samples,
    read_training_log,
    run_detail,
    run_dir,
)


ensure_demo_run()

app = FastAPI(title="MLX GPT Lab Dashboard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def not_found(exc: Exception) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


def probe_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/api/health")
def health():
    runs_dir = get_runs_dir()
    return {"ok": True, "service": "mlx-gpt-lab-dashboard-api", "runs_dir": str(runs_dir), "runs_dir_exists": runs_dir.exists()}


@app.get("/api/system-resources")
def system_resources():
    return get_system_resources()


@app.get("/api/runs")
def runs():
    return {"runs_dir": str(get_runs_dir()), "runs": list_runs()}


@app.get("/api/runs/{run_id}")
def run(run_id: str):
    try:
        return run_detail(run_id)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc


@app.get("/api/runs/{run_id}/config")
def config(run_id: str):
    try:
        return {"run_id": run_id, "config": get_config(run_id)}
    except FileNotFoundError as exc:
        raise not_found(exc) from exc


@app.get("/api/runs/{run_id}/training-log")
def training_log(
    run_id: str,
    tail: int | None = None,
    heartbeat_tail: int | None = None,
    heartbeat_points: int | None = 900,
):
    try:
        return {
            "run_id": run_id,
            "rows": read_training_log(
                run_dir(run_id),
                tail=tail,
                heartbeat_tail=heartbeat_tail,
                heartbeat_points=heartbeat_points,
            ),
        }
    except FileNotFoundError as exc:
        raise not_found(exc) from exc


@app.get("/api/runs/{run_id}/metrics")
def metrics(run_id: str):
    try:
        return {"run_id": run_id, "metrics": get_metrics(run_id)}
    except FileNotFoundError as exc:
        raise not_found(exc) from exc


@app.get("/api/runs/{run_id}/status")
def status(run_id: str):
    try:
        return get_status(run_id)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc


@app.get("/api/runs/{run_id}/samples")
def samples(run_id: str):
    try:
        return {"run_id": run_id, "samples": list_samples(run_dir(run_id))}
    except FileNotFoundError as exc:
        raise not_found(exc) from exc


@app.get("/api/runs/{run_id}/samples/{sample_name}")
def sample(run_id: str, sample_name: str):
    try:
        text = get_sample_text(run_dir(run_id), sample_name)
        if text is None:
            raise FileNotFoundError(f"sample not found: {sample_name}")
        return {"run_id": run_id, "sample_name": sample_name, "text": text}
    except FileNotFoundError as exc:
        raise not_found(exc) from exc


@app.get("/api/runs/{run_id}/dataset-samples")
def dataset_samples(
    run_id: str,
    limit: int = 12,
    mode: str = "random",
    seed: int | None = None,
    scan_limit: int = 200000,
    include_tokens: bool = True,
    token_limit: int = 80,
):
    try:
        return get_dataset_samples(
            run_id,
            limit=limit,
            mode=mode,
            seed=seed,
            scan_limit=scan_limit,
            include_tokens=include_tokens,
            token_limit=token_limit,
        )
    except FileNotFoundError as exc:
        raise not_found(exc) from exc


@app.get("/api/runs/{run_id}/tokenizer-vocab")
def tokenizer_vocab(run_id: str, page: int = 1, page_size: int = 256, query: str | None = None):
    try:
        return get_tokenizer_vocab(run_id, page=page, page_size=page_size, query=query)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc


@app.get("/api/runs/{run_id}/final-text")
def final_text(run_id: str):
    try:
        return get_final_text(run_id)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc


@app.get("/api/runs/{run_id}/eval-results")
def eval_results(run_id: str):
    try:
        return get_eval_results(run_id)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc


@app.get("/api/runs/{run_id}/benchmark")
def benchmark(run_id: str):
    try:
        return get_benchmark(run_id)
    except FileNotFoundError as exc:
        raise not_found(exc) from exc


@app.get("/api/runs/{run_id}/checkpoints")
def checkpoints(run_id: str):
    try:
        return {"run_id": run_id, "checkpoints": list_checkpoints(run_dir(run_id))}
    except FileNotFoundError as exc:
        raise not_found(exc) from exc


@app.post("/api/probe/tokenize")
def probe_tokenize(payload: dict):
    try:
        return tokenize_prompt(
            str(payload.get("run_id", "")),
            str(payload.get("checkpoint_name", "")),
            str(payload.get("prompt", "")),
            max_context_tokens=payload.get("max_context_tokens"),
        )
    except Exception as exc:
        raise probe_error(exc) from exc


@app.post("/api/probe/next-token")
def probe_next_token(payload: dict):
    try:
        return next_token_probe(
            str(payload.get("run_id", "")),
            str(payload.get("checkpoint_name", "")),
            str(payload.get("prompt", "")),
            top_k=int(payload.get("top_k", 20)),
            temperature=float(payload.get("temperature", 1.0)),
            max_context_tokens=payload.get("max_context_tokens"),
            temperature_values=payload.get("temperature_values"),
        )
    except Exception as exc:
        raise probe_error(exc) from exc


@app.post("/api/probe/generation-trace")
def probe_generation_trace(payload: dict):
    try:
        return generation_trace_probe(
            str(payload.get("run_id", "")),
            str(payload.get("checkpoint_name", "")),
            str(payload.get("prompt", "")),
            steps=int(payload.get("steps", 8)),
            top_k=int(payload.get("top_k", 10)),
            temperature=float(payload.get("temperature", 1.0)),
            max_context_tokens=payload.get("max_context_tokens"),
        )
    except Exception as exc:
        raise probe_error(exc) from exc


@app.post("/api/probe/checkpoint-compare")
def probe_checkpoint_compare(payload: dict):
    try:
        return checkpoint_compare_probe(
            str(payload.get("run_id", "")),
            list(payload.get("checkpoint_names", [])),
            str(payload.get("prompt", "")),
            top_k=int(payload.get("top_k", 12)),
            temperature=float(payload.get("temperature", 1.0)),
            max_context_tokens=payload.get("max_context_tokens"),
        )
    except Exception as exc:
        raise probe_error(exc) from exc


@app.post("/api/probe/token-neighborhood")
def probe_token_neighborhood(payload: dict):
    try:
        return token_neighborhood_probe(
            str(payload.get("run_id", "")),
            str(payload.get("checkpoint_name", "")),
            str(payload.get("query", "")),
            space=str(payload.get("space", "embedding")),
            top_k=int(payload.get("top_k", 20)),
            include_self=bool(payload.get("include_self", False)),
            prompt=str(payload.get("prompt", "")),
            temperature=float(payload.get("temperature", 1.0)),
            max_context_tokens=payload.get("max_context_tokens"),
            max_targets=int(payload.get("max_targets", 8)),
        )
    except Exception as exc:
        raise probe_error(exc) from exc


@app.post("/api/probe/attention")
def probe_attention(payload: dict):
    try:
        return attention_probe(
            str(payload.get("run_id", "")),
            str(payload.get("checkpoint_name", "")),
            str(payload.get("prompt", "")),
            layer=int(payload.get("layer", 0)),
            head=int(payload.get("head", 0)),
            max_context_tokens=payload.get("max_context_tokens"),
        )
    except Exception as exc:
        raise probe_error(exc) from exc


@app.post("/api/probe/unload")
def probe_unload():
    return unload_probe()
