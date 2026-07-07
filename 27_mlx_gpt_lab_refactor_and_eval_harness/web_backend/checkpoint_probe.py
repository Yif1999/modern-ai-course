from __future__ import annotations

import math
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from run_reader import dataset_metadata_for_run, read_json, resolve_data_path, run_dir, safe_child


@dataclass
class LoadedProbe:
    key: tuple[str, str]
    model: Any
    tokenizer: Any
    config: dict[str, Any]
    block_size: int
    dtype: str
    vector_cache: dict[str, Any] = field(default_factory=dict)


_LOADED_PROBE: LoadedProbe | None = None


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def _dtype_from_name(name: str):
    return {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }.get(name, mx.bfloat16)


def _load_qwen_like_model(config: dict[str, Any], course_dir: Path):
    model_type = str(config.get("model_type", ""))
    if model_type in {"baseline_debug", "qwen_dense_tiny"}:
        src_dir = course_dir / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        from mlx_gpt_lab.model_factory import create_model

        return create_model(config, int(config["vocab_size"]))

    src_dir = course_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from model_qwen_like import QwenLikeConfig, QwenLikeDenseLM

    model_cfg = QwenLikeConfig(
        vocab_size=int(config["vocab_size"]),
        block_size=int(config["block_size"]),
        n_embd=int(config["n_embd"]),
        num_layers=int(config["num_layers"]),
        num_q_heads=int(config["num_q_heads"]),
        num_kv_heads=int(config["num_kv_heads"]),
        ffn_multiplier=float(config.get("ffn_multiplier", 3.0)),
        rope_base=float(config.get("rope_base", 1_000_000.0)),
        qk_norm=bool(config.get("qk_norm", True)),
        weight_tying=bool(config.get("weight_tying", True)),
        use_bias=bool(config.get("use_bias", False)),
        activation_checkpointing=False,
        rope_impl=str(config.get("rope_impl", "fast")),
        rms_norm_impl=str(config.get("rms_norm_impl", "fast")),
    )
    model = QwenLikeDenseLM(model_cfg)
    dtype = str(config.get("dtype", "bfloat16"))
    if dtype != "float32":
        model.set_dtype(_dtype_from_name(dtype))
    return model


def _tokenizer_for_run(path: Path, config: dict[str, Any]) -> Any:
    _, metadata = dataset_metadata_for_run(path)
    tokenizer_path = (
        resolve_data_path(config.get("tokenizer_path"), path)
        or resolve_data_path(metadata.get("tokenizer_path"), path)
    )
    if not tokenizer_path or not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer_path not found: {tokenizer_path}")

    tokenizer_type = str(config.get("tokenizer_type") or metadata.get("tokenizer_type") or "").lower()
    if tokenizer_type in {"char", "bpe"}:
        course_dir = path.parents[2]
        src_dir = course_dir / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        from mlx_gpt_lab.tokenizer import load_tokenizer

        return load_tokenizer(tokenizer_path, tokenizer_type)

    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_path))


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    ids = getattr(encoded, "ids", encoded)
    return [int(item) for item in ids]


def _decode_ids(tokenizer: Any, ids: list[int], *, skip_special_tokens: bool) -> str:
    return tokenizer.decode([int(item) for item in ids], skip_special_tokens=skip_special_tokens)


def unload_probe() -> dict[str, Any]:
    global _LOADED_PROBE
    previous = _LOADED_PROBE.key if _LOADED_PROBE else None
    _LOADED_PROBE = None
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    return {"ok": True, "unloaded": previous}


def _token_info(tokenizer: Any, token_id: int) -> dict[str, Any]:
    token_id = int(token_id)
    try:
        decoded_raw = _decode_ids(tokenizer, [token_id], skip_special_tokens=False)
    except Exception:
        decoded_raw = ""
    display, readable, kind = _display_decoded_token(decoded_raw)
    return {
        "id": token_id,
        "decoded": display,
        "display": display,
        "readable": readable,
        "kind": kind,
        "raw_decoded": decoded_raw,
    }


def _resolve_target_ids(tokenizer: Any, query: str, vocab_size: int, max_targets: int) -> tuple[list[int], str]:
    text = str(query or "").strip()
    if not text:
        raise ValueError("请输入 token 文本或 token id。")

    normalized = text.removeprefix("id:").removeprefix("#").strip()
    if normalized.isdigit():
        token_id = int(normalized)
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(f"token id 超出词表范围: {token_id}")
        return [token_id], "id"

    ids = _encode_ids(tokenizer, text)
    if not ids:
        raise ValueError("query 无法被 tokenizer 编码。")
    return [int(item) for item in ids[:max_targets]], "text"


def _space_weights(probe: LoadedProbe, space: str):
    space = space if space in {"embedding", "lm_head"} else "embedding"
    configured_weight_tying = probe.config.get("weight_tying")
    weight_tying = bool(configured_weight_tying) if configured_weight_tying is not None else not hasattr(probe.model, "lm_head")
    vocab_size = int(probe.config.get("vocab_size", 0))

    if space == "embedding" or weight_tying:
        if hasattr(probe.model, "token_embedding"):
            return probe.model.token_embedding, "embedding", weight_tying
        if hasattr(probe.model, "token_embedding_table"):
            return probe.model.token_embedding_table, "embedding", weight_tying
        raise ValueError("当前模型没有可识别的 token embedding 权重。")

    lm_head = getattr(probe.model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if weight is None:
        raise ValueError("当前模型没有独立 lm_head 权重。")
    if len(weight.shape) == 2 and int(weight.shape[0]) != vocab_size and int(weight.shape[1]) == vocab_size:
        weight = mx.transpose(weight, (1, 0))
    return weight, "lm_head", weight_tying


def _normalized_space(probe: LoadedProbe, space: str):
    weights, effective_space, weight_tying = _space_weights(probe, space)
    cache_key = f"{effective_space}:normalized"
    if cache_key not in probe.vector_cache:
        matrix = weights.astype(mx.float32)
        norms = mx.sqrt(mx.sum(matrix * matrix, axis=1, keepdims=True))
        normalized = matrix / mx.maximum(norms, mx.array(1e-12, dtype=mx.float32))
        mx.eval(normalized)
        probe.vector_cache[cache_key] = normalized
    return probe.vector_cache[cache_key], effective_space, weight_tying


def _neighbor_rows(
    probe: LoadedProbe,
    normalized,
    target_id: int,
    *,
    top_k: int,
    include_self: bool,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    vocab_size = int(normalized.shape[0])
    if target_id < 0 or target_id >= vocab_size:
        raise ValueError(f"token id 超出词表范围: {target_id}")

    sims = normalized @ normalized[target_id]
    mx.eval(sims)
    sims_np = np.asarray(sims, dtype=np.float32)
    if not include_self:
        sims_np[target_id] = -np.inf

    take = min(max(int(top_k), 1), max(1, vocab_size - (0 if include_self else 1)))
    subset = np.argpartition(-sims_np, take - 1)[:take]
    subset = subset[np.argsort(-sims_np[subset])]
    rows = []
    for rank, token_id in enumerate(subset, start=1):
        similarity = float(sims_np[token_id])
        rows.append(
            {
                "rank": rank,
                **_token_info(probe.tokenizer, int(token_id)),
                "similarity": similarity,
                "distance": float(1.0 - similarity),
                "is_self": int(token_id) == int(target_id),
            }
        )
    return rows, subset.astype(np.int64)


def _plot_points(
    probe: LoadedProbe,
    normalized,
    target_id: int,
    neighbor_ids: np.ndarray,
    neighbors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ids = np.array([int(target_id), *[int(item) for item in neighbor_ids if int(item) != int(target_id)]], dtype=np.int64)
    ids = ids[:41]
    if ids.size == 0:
        return []
    neighbor_meta = {int(item["id"]): item for item in neighbors}
    vectors_mx = normalized[mx.array(ids.tolist(), dtype=mx.int32)]
    mx.eval(vectors_mx)
    vectors = np.asarray(vectors_mx, dtype=np.float32)
    if vectors.shape[0] == 1:
        coords = np.zeros((1, 2), dtype=np.float32)
    else:
        centered = vectors - vectors.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            coords = centered @ vt[:2].T
            if coords.shape[1] == 1:
                coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)], axis=1)
        except np.linalg.LinAlgError:
            coords = np.zeros((vectors.shape[0], 2), dtype=np.float32)

    max_abs = float(np.max(np.abs(coords))) if coords.size else 0.0
    if max_abs > 0:
        coords = coords / max_abs

    points = []
    for index, token_id in enumerate(ids):
        meta = neighbor_meta.get(int(token_id), {})
        points.append(
            {
                **_token_info(probe.tokenizer, int(token_id)),
                "x": float(coords[index, 0]),
                "y": float(coords[index, 1]),
                "rank": int(meta.get("rank", 0)),
                "similarity": float(meta.get("similarity", 1.0 if int(token_id) == int(target_id) else 0.0)),
                "distance": float(meta.get("distance", 0.0 if int(token_id) == int(target_id) else 1.0)),
                "is_target": int(token_id) == int(target_id),
            }
        )
    return points


def _target_generation_rows(
    probe: LoadedProbe,
    prompt: str,
    target_ids: list[int],
    *,
    temperature: float,
    max_context_tokens: int | None,
) -> dict[str, Any] | None:
    prompt = str(prompt or "")
    if not prompt.strip():
        return None
    ids, context_ids, logits = _forward_logits(probe, prompt, max_context_tokens=max_context_tokens)
    probs = _softmax_np(logits, temperature)
    rows = []
    for token_id in target_ids:
        token_id = int(token_id)
        rows.append(
            {
                **_token_info(probe.tokenizer, token_id),
                "rank": int(np.count_nonzero(logits > logits[token_id]) + 1),
                "logit": float(logits[token_id]),
                "probability": float(probs[token_id]),
            }
        )
    top_candidates, stats = _top_candidates(
        probe.tokenizer,
        logits,
        context_ids,
        top_k=8,
        temperature=temperature,
    )
    return {
        "prompt": prompt,
        "token_count": len(ids),
        "context_token_count": len(context_ids),
        "context_truncated": len(ids) > len(context_ids),
        "temperature": float(temperature),
        "targets": rows,
        "top_candidates": top_candidates,
        "stats": stats,
    }


def _checkpoint_path(path: Path, checkpoint_name: str) -> Path:
    ckpt_dir = path / "checkpoints"
    ckpt_path = safe_child(ckpt_dir, checkpoint_name)
    if not ckpt_path.exists() or not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_name}")
    if not ckpt_path.name.endswith("_model.safetensors"):
        raise FileNotFoundError("请选择 *_model.safetensors 权重文件。")
    return ckpt_path


def load_probe(run_id: str, checkpoint_name: str) -> LoadedProbe:
    global _LOADED_PROBE
    key = (run_id, checkpoint_name)
    if _LOADED_PROBE is not None and _LOADED_PROBE.key == key:
        return _LOADED_PROBE

    path = run_dir(run_id)
    config = read_json(path / "config.json")
    if not isinstance(config, dict):
        raise FileNotFoundError("config.json not found or invalid")
    model_type = str(config.get("model_type", ""))
    if model_type not in {"qwen_like_dense_maxout", "qwen_dense_tiny", "baseline_debug", ""}:
        raise ValueError(f"Unsupported probe model_type: {model_type}")

    course_dir = path.parents[2]
    model = _load_qwen_like_model(config, course_dir)
    ckpt_path = _checkpoint_path(path, checkpoint_name)
    model.load_weights(str(ckpt_path))
    tokenizer = _tokenizer_for_run(path, config)
    mx.eval(model.parameters())

    _LOADED_PROBE = LoadedProbe(
        key=key,
        model=model,
        tokenizer=tokenizer,
        config=config,
        block_size=int(config["block_size"]),
        dtype=str(config.get("dtype", "bfloat16")),
    )
    return _LOADED_PROBE


def _encoding_preview(tokenizer: Any, ids: list[int]) -> list[dict[str, Any]]:
    rows = []
    for index, token_id in enumerate(ids):
        try:
            decoded = _decode_ids(tokenizer, [int(token_id)], skip_special_tokens=False)
        except Exception:
            decoded = ""
        display, readable, kind = _display_decoded_token(decoded)
        rows.append(
            {
                "index": index,
                "id": int(token_id),
                "decoded": display,
                "display": display,
                "readable": readable,
                "kind": kind,
                "raw_decoded": decoded,
            }
        )
    return rows


def _visible_token_text(value: str) -> str:
    if value == "":
        return "∅"
    return value.replace(" ", "·").replace("\n", "↵").replace("\t", "⇥").replace("\r", "␍")


def _display_decoded_token(value: str) -> tuple[str, bool, str]:
    if value == "":
        return "∅", True, "empty"
    if "\ufffd" in value:
        return "byte", False, "byte_fragment"
    if all(unicodedata.category(char).startswith("C") for char in value):
        return "ctrl", False, "control"
    return _visible_token_text(value), True, "text"


def _softmax_np(logits: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1e-6)
    scaled = logits.astype(np.float64) / temperature
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    return exp / np.sum(exp)


def _top_candidates(
    tokenizer: Any,
    logits: np.ndarray,
    context_ids: list[int],
    *,
    top_k: int,
    temperature: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    probs = _softmax_np(logits, temperature)
    top_k = max(1, min(int(top_k), 100))
    order = np.argsort(-probs)[:top_k]
    context_tail = set(int(item) for item in context_ids[-128:])
    candidates = []
    for rank, token_id in enumerate(order, start=1):
        token_id_int = int(token_id)
        try:
            decoded_raw = _decode_ids(tokenizer, [token_id_int], skip_special_tokens=False)
        except Exception:
            decoded_raw = ""
        display, readable, kind = _display_decoded_token(decoded_raw)
        candidates.append(
            {
                "rank": rank,
                "id": token_id_int,
                "decoded": display,
                "display": display,
                "readable": readable,
                "kind": kind,
                "raw_decoded": decoded_raw,
                "logit": float(logits[token_id_int]),
                "probability": float(probs[token_id_int]),
                "repeats_recent_context": token_id_int in context_tail,
            }
        )
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-40))))
    top_prob = float(candidates[0]["probability"]) if candidates else None
    repetition_count = sum(1 for item in candidates if item["repeats_recent_context"])
    stats = {
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(len(probs)) if len(probs) > 1 else 0.0,
        "top_probability": top_prob,
        "repetition_candidates": repetition_count,
        "repetition_ratio": repetition_count / len(candidates) if candidates else 0.0,
    }
    return candidates, stats


def _forward_logits(probe: LoadedProbe, prompt: str, max_context_tokens: int | None = None):
    ids = _encode_ids(probe.tokenizer, prompt)
    max_context = max(1, min(int(max_context_tokens or probe.block_size), probe.block_size))
    context_ids = ids[-max_context:]
    if not context_ids:
        raise ValueError("prompt 为空，至少输入一个 token。")
    idx = mx.array([context_ids], dtype=mx.int32)
    logits = probe.model(idx)[0, -1].astype(mx.float32)
    mx.eval(logits)
    logits_np = np.asarray(logits, dtype=np.float32)
    return ids, context_ids, logits_np


def tokenize_prompt(run_id: str, checkpoint_name: str, prompt: str, max_context_tokens: int | None = None):
    probe = load_probe(run_id, checkpoint_name)
    ids = _encode_ids(probe.tokenizer, prompt)
    max_context = max(1, min(int(max_context_tokens or probe.block_size), probe.block_size))
    context_ids = ids[-max_context:]
    return _json_safe(
        {
            "run_id": run_id,
            "checkpoint_name": checkpoint_name,
            "prompt": prompt,
            "token_count": len(ids),
            "context_token_count": len(context_ids),
            "context_truncated": len(ids) > len(context_ids),
            "block_size": probe.block_size,
            "tokens": _encoding_preview(probe.tokenizer, context_ids),
        }
    )


def next_token_probe(
    run_id: str,
    checkpoint_name: str,
    prompt: str,
    *,
    top_k: int = 20,
    temperature: float = 1.0,
    max_context_tokens: int | None = None,
    temperature_values: list[float] | None = None,
):
    probe = load_probe(run_id, checkpoint_name)
    ids, context_ids, logits = _forward_logits(probe, prompt, max_context_tokens=max_context_tokens)
    candidates, stats = _top_candidates(
        probe.tokenizer,
        logits,
        context_ids,
        top_k=top_k,
        temperature=temperature,
    )
    temp_views = []
    for temp in temperature_values or [0.5, 0.8, 1.0, 1.5]:
        view_candidates, view_stats = _top_candidates(
            probe.tokenizer,
            logits,
            context_ids,
            top_k=min(top_k, 12),
            temperature=float(temp),
        )
        temp_views.append({"temperature": float(temp), "candidates": view_candidates, "stats": view_stats})
    return _json_safe(
        {
            "run_id": run_id,
            "checkpoint_name": checkpoint_name,
            "prompt": prompt,
            "token_count": len(ids),
            "context_token_count": len(context_ids),
            "context_truncated": len(ids) > len(context_ids),
            "block_size": probe.block_size,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "tokenization": _encoding_preview(probe.tokenizer, context_ids),
            "candidates": candidates,
            "stats": stats,
            "temperature_views": temp_views,
        }
    )


def generation_trace_probe(
    run_id: str,
    checkpoint_name: str,
    prompt: str,
    *,
    steps: int = 8,
    top_k: int = 10,
    temperature: float = 1.0,
    max_context_tokens: int | None = None,
):
    probe = load_probe(run_id, checkpoint_name)
    ids = _encode_ids(probe.tokenizer, prompt)
    if not ids:
        raise ValueError("prompt 为空，至少输入一个 token。")
    max_context = max(1, min(int(max_context_tokens or probe.block_size), probe.block_size))
    trace = []
    steps = max(1, min(int(steps), 64))
    for step in range(steps):
        context_ids = ids[-max_context:]
        idx = mx.array([context_ids], dtype=mx.int32)
        logits = probe.model(idx)[0, -1].astype(mx.float32)
        mx.eval(logits)
        logits_np = np.asarray(logits, dtype=np.float32)
        candidates, stats = _top_candidates(
            probe.tokenizer,
            logits_np,
            context_ids,
            top_k=top_k,
            temperature=temperature,
        )
        selected = candidates[0]
        ids.append(int(selected["id"]))
        trace.append(
            {
                "step": step,
                "selected": selected,
                "candidates": candidates,
                "stats": stats,
                "context_tail": _encoding_preview(probe.tokenizer, context_ids[-16:]),
            }
        )
    generated_text = _decode_ids(probe.tokenizer, ids, skip_special_tokens=True)
    return _json_safe(
        {
            "run_id": run_id,
            "checkpoint_name": checkpoint_name,
            "prompt": prompt,
            "steps": steps,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "generated_text": generated_text,
            "trace": trace,
        }
    )


def _apply_probe_rope(attn: Any, q, k):
    rope_impl = str(getattr(attn, "rope_impl", "manual"))
    head_dim = int(getattr(attn, "head_dim"))
    rope_base = float(getattr(attn, "rope_base", 1_000_000.0))
    if rope_impl == "fast":
        return (
            mx.fast.rope(q, dims=head_dim, traditional=False, base=rope_base, scale=1.0, offset=0),
            mx.fast.rope(k, dims=head_dim, traditional=False, base=rope_base, scale=1.0, offset=0),
        )
    if rope_impl == "nn" and getattr(attn, "rope", None) is not None:
        return attn.rope(q), attn.rope(k)

    # Manual fallback mirrors the teaching model implementation. Keep it local
    # so the dashboard can inspect checkpoints without changing training code.
    _, _, seq_len, _ = q.shape
    inv_freq = 1.0 / (rope_base ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
    positions = mx.arange(seq_len).astype(mx.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    cos = mx.cos(freqs).reshape(1, 1, seq_len, head_dim // 2)
    sin = mx.sin(freqs).reshape(1, 1, seq_len, head_dim // 2)
    cos = mx.stack((cos, cos), axis=-1).reshape(1, 1, seq_len, head_dim)
    sin = mx.stack((sin, sin), axis=-1).reshape(1, 1, seq_len, head_dim)

    def rotate_half_interleaved(x):
        even = x[..., 0::2]
        odd = x[..., 1::2]
        return mx.stack((-odd, even), axis=-1).reshape(x.shape)

    return q * cos + rotate_half_interleaved(q) * sin, k * cos + rotate_half_interleaved(k) * sin


def _masked_softmax_np(scores: np.ndarray) -> np.ndarray:
    seq_len = scores.shape[-1]
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    masked = scores.astype(np.float64).copy()
    masked[mask] = -np.inf
    row_max = np.max(masked, axis=-1, keepdims=True)
    exp = np.exp(masked - row_max)
    exp[mask] = 0.0
    denom = np.sum(exp, axis=-1, keepdims=True)
    return exp / np.maximum(denom, 1e-40)


def _attention_head_summary(tokenizer: Any, weights: np.ndarray, tokens: list[dict[str, Any]], head: int, kv_head: int):
    last_row = weights[-1]
    order = np.argsort(-last_row)[: min(8, len(last_row))]
    entropy = float(-np.sum(last_row * np.log(np.maximum(last_row, 1e-40))))
    return {
        "head": int(head),
        "kv_head": int(kv_head),
        "last_token_entropy": entropy,
        "last_token_top": [
            {
                "position": int(pos),
                "weight": float(last_row[pos]),
                "token": tokens[int(pos)],
            }
            for pos in order
        ],
    }


def attention_probe(
    run_id: str,
    checkpoint_name: str,
    prompt: str,
    *,
    layer: int = 0,
    head: int = 0,
    max_context_tokens: int | None = None,
):
    probe = load_probe(run_id, checkpoint_name)
    ids = _encode_ids(probe.tokenizer, prompt)
    if not ids:
        raise ValueError("prompt 为空，至少输入一个 token。")

    # Keep visualization bounded. The training context can be 1024, but a
    # 1024x1024 heatmap is neither readable nor cheap for a dashboard.
    requested_context = int(max_context_tokens or 64)
    max_context = max(2, min(requested_context, probe.block_size, 192))
    context_ids = ids[-max_context:]
    idx = mx.array([context_ids], dtype=mx.int32)

    model = probe.model
    blocks = getattr(model, "blocks", None)
    if not blocks:
        raise ValueError("当前模型不支持 attention inspection：没有 blocks。")
    num_layers = len(blocks)
    layer = max(0, min(int(layer), num_layers - 1))

    x = model.token_embedding[idx]
    for block_index in range(layer):
        x = blocks[block_index](x)

    block = blocks[layer]
    attn = block.attn
    x_norm = block.input_norm(x)
    q = attn.split_heads(attn.q_proj(x_norm), attn.num_q_heads)
    k = attn.split_heads(attn.k_proj(x_norm), attn.num_kv_heads)
    q, k = _apply_probe_rope(attn, q, k)
    if getattr(attn, "q_norm", None) is not None:
        q = attn.q_norm(q)
        k = attn.k_norm(k)
    q = q.astype(mx.float32)
    k = k.astype(mx.float32)
    mx.eval(q, k)

    q_np = np.array(q, dtype=np.float32)[0]
    k_np = np.array(k, dtype=np.float32)[0]
    num_q_heads = int(q_np.shape[0])
    num_kv_heads = int(k_np.shape[0])
    head_dim = int(q_np.shape[-1])
    head = max(0, min(int(head), num_q_heads - 1))
    kv_group_size = max(1, num_q_heads // max(1, num_kv_heads))

    tokens = _encoding_preview(probe.tokenizer, context_ids)
    selected_kv_head = min(num_kv_heads - 1, head // kv_group_size)
    scores = (q_np[head] @ np.swapaxes(k_np[selected_kv_head], -1, -2)) / math.sqrt(head_dim)
    weights = _masked_softmax_np(scores)

    head_summaries = []
    for h in range(num_q_heads):
        kv_h = min(num_kv_heads - 1, h // kv_group_size)
        h_scores = (q_np[h] @ np.swapaxes(k_np[kv_h], -1, -2)) / math.sqrt(head_dim)
        h_weights = _masked_softmax_np(h_scores)
        head_summaries.append(_attention_head_summary(probe.tokenizer, h_weights, tokens, h, kv_h))

    # Reduce JSON noise but preserve enough precision for color scales and top
    # attended token inspection.
    rounded_weights = [[round(float(value), 6) for value in row] for row in weights]
    return _json_safe(
        {
            "run_id": run_id,
            "checkpoint_name": checkpoint_name,
            "prompt": prompt,
            "layer": layer,
            "head": head,
            "kv_head": selected_kv_head,
            "num_layers": num_layers,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "kv_group_size": kv_group_size,
            "token_count": len(ids),
            "context_token_count": len(context_ids),
            "context_truncated": len(ids) > len(context_ids),
            "block_size": probe.block_size,
            "tokens": tokens,
            "weights": rounded_weights,
            "selected_head_summary": _attention_head_summary(probe.tokenizer, weights, tokens, head, selected_kv_head),
            "head_summaries": head_summaries,
            "notes": [
                "Heatmap 行表示 query 位置，列表示 key 位置。",
                "使用 causal mask，未来 token 权重固定为 0。",
                "当前模型使用 GQA：多个 Q heads 共享较少 K/V heads。",
            ],
        }
    )


def token_neighborhood_probe(
    run_id: str,
    checkpoint_name: str,
    query: str,
    *,
    space: str = "embedding",
    top_k: int = 20,
    include_self: bool = False,
    prompt: str = "",
    temperature: float = 1.0,
    max_context_tokens: int | None = None,
    max_targets: int = 8,
):
    probe = load_probe(run_id, checkpoint_name)
    top_k = max(1, min(int(top_k), 80))
    max_targets = max(1, min(int(max_targets), 8))
    vocab_size = int(probe.config.get("vocab_size", 0))
    target_ids, query_mode = _resolve_target_ids(probe.tokenizer, query, vocab_size, max_targets)
    normalized, effective_space, weight_tying = _normalized_space(probe, space)

    targets = []
    for target_id in target_ids:
        neighbors, neighbor_ids = _neighbor_rows(
            probe,
            normalized,
            int(target_id),
            top_k=top_k,
            include_self=include_self,
        )
        targets.append(
            {
                **_token_info(probe.tokenizer, int(target_id)),
                "neighbors": neighbors,
                "plot_points": _plot_points(probe, normalized, int(target_id), neighbor_ids[:40], neighbors),
            }
        )

    generation = _target_generation_rows(
        probe,
        prompt,
        target_ids,
        temperature=temperature,
        max_context_tokens=max_context_tokens,
    )

    return _json_safe(
        {
            "run_id": run_id,
            "checkpoint_name": checkpoint_name,
            "query": query,
            "query_mode": query_mode,
            "requested_space": space,
            "space": effective_space,
            "weight_tying": weight_tying,
            "top_k": top_k,
            "include_self": bool(include_self),
            "tokenization": _encoding_preview(probe.tokenizer, target_ids),
            "targets": targets,
            "generation": generation,
            "notes": {
                "space": (
                    "weight_tying=true，输出侧 lm_head 与输入 embedding 共用同一张矩阵。"
                    if weight_tying
                    else "embedding 与 lm_head 可分别查看；lm_head 更接近生成端 token 竞争关系。"
                ),
                "cache": f"{effective_space} normalized matrix is cached for the loaded checkpoint.",
            },
        }
    )


def checkpoint_compare_probe(
    run_id: str,
    checkpoint_names: list[str],
    prompt: str,
    *,
    top_k: int = 12,
    temperature: float = 1.0,
    max_context_tokens: int | None = None,
):
    names = [name for name in checkpoint_names if name.endswith("_model.safetensors")][:4]
    if not names:
        raise ValueError("请选择至少一个 *_model.safetensors checkpoint。")
    rows = []
    for name in names:
        result = next_token_probe(
            run_id,
            name,
            prompt,
            top_k=top_k,
            temperature=temperature,
            max_context_tokens=max_context_tokens,
            temperature_values=[temperature],
        )
        rows.append(
            {
                "checkpoint_name": name,
                "candidates": result["candidates"],
                "stats": result["stats"],
            }
        )
    return _json_safe(
        {
            "run_id": run_id,
            "prompt": prompt,
            "top_k": int(top_k),
            "temperature": float(temperature),
            "comparisons": rows,
        }
    )
