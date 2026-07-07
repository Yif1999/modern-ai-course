from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from mlx.utils import tree_flatten, tree_map, tree_unflatten

CURRENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CURRENT_DIR / "src"))

from model_qwen_like import QwenLikeConfig, QwenLikeDenseLM  # noqa: E402
from telemetry import telemetry_snapshot, write_json  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def count_parameters(params) -> int:
    total = 0
    for _, value in tree_flatten(params):
        if hasattr(value, "shape"):
            total += int(np.prod(value.shape))
    return total


def tree_l2_norm(tree):
    norm_sq = 0.0
    for _, value in tree_flatten(tree):
        if hasattr(value, "astype"):
            v = value.astype(mx.float32)
            norm_sq = norm_sq + mx.sum(v * v)
    return mx.sqrt(norm_sq)


def dtype_from_name(name: str):
    return {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }[name]


def pair_from_config(value: Any, default: tuple[float, float]) -> list[float]:
    if value is None:
        return [float(default[0]), float(default[1])]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Expected a pair of floats, got: {value!r}")
    return [float(value[0]), float(value[1])]


def canonical_lr_schedule_name(cfg: dict[str, Any]) -> str:
    name = str(cfg.get("lr_schedule", cfg.get("learning_rate_schedule", "constant"))).strip().lower()
    aliases = {
        "none": "constant",
        "fixed": "constant",
        "constant": "constant",
        "cosine": "cosine",
        "warmup-cosine": "warmup_cosine",
        "warmup_cosine": "warmup_cosine",
        "linear-warmup-cosine": "warmup_cosine",
        "linear_warmup_cosine": "warmup_cosine",
        "resume-ramp-cosine": "resume_ramp_cosine",
        "resume_ramp_cosine": "resume_ramp_cosine",
        "warmup-linear": "warmup_linear",
        "warmup_linear": "warmup_linear",
        "linear": "linear",
    }
    if name not in aliases:
        raise ValueError(
            f"Unsupported lr_schedule: {cfg.get('lr_schedule')!r}. "
            "Use constant, cosine, warmup_cosine, linear, or warmup_linear."
        )
    return aliases[name]


def lr_schedule_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    name = canonical_lr_schedule_name(cfg)
    max_lr = float(cfg["learning_rate"])
    min_lr = float(cfg.get("min_learning_rate", cfg.get("min_lr", 0.0)))
    warmup_steps = int(cfg.get("warmup_steps", 0))
    warmup_init_lr = float(cfg.get("warmup_init_lr", 0.0))
    decay_steps = int(
        cfg.get(
            "lr_decay_steps",
            cfg.get("total_training_steps", cfg.get("max_iters", 1)),
        )
    )
    decay_steps = max(1, decay_steps)
    return {
        "lr_schedule": name,
        "max_learning_rate": max_lr,
        "min_learning_rate": min_lr,
        "warmup_steps": warmup_steps,
        "warmup_init_lr": warmup_init_lr,
        "lr_decay_steps": decay_steps,
        "resume_ramp_start_step": int(cfg.get("resume_ramp_start_step", cfg.get("lr_ramp_start_step", 0))),
        "resume_ramp_steps": int(cfg.get("resume_ramp_steps", cfg.get("lr_ramp_steps", 0))),
        "resume_ramp_init_lr": float(cfg.get("resume_ramp_init_lr", cfg.get("lr_ramp_init_lr", warmup_init_lr))),
    }


def build_learning_rate(cfg: dict[str, Any]):
    meta = lr_schedule_metadata(cfg)
    name = meta["lr_schedule"]
    max_lr = meta["max_learning_rate"]
    min_lr = meta["min_learning_rate"]
    warmup_steps = meta["warmup_steps"]
    warmup_init_lr = meta["warmup_init_lr"]
    decay_steps = meta["lr_decay_steps"]

    if name == "constant":
        return max_lr

    if name == "linear":
        return optim.linear_schedule(max_lr, min_lr, decay_steps)

    if name == "cosine":
        return optim.cosine_decay(max_lr, decay_steps, end=min_lr)

    if name == "warmup_linear":
        if warmup_steps < 1:
            return optim.linear_schedule(max_lr, min_lr, decay_steps)
        warmup = optim.linear_schedule(warmup_init_lr, max_lr, warmup_steps)
        linear = optim.linear_schedule(max_lr, min_lr, max(1, decay_steps - warmup_steps))
        return optim.join_schedules([warmup, linear], [warmup_steps])

    if name == "warmup_cosine":
        if warmup_steps < 1:
            return optim.cosine_decay(max_lr, decay_steps, end=min_lr)
        warmup = optim.linear_schedule(warmup_init_lr, max_lr, warmup_steps)
        cosine = optim.cosine_decay(max_lr, max(1, decay_steps - warmup_steps), end=min_lr)
        return optim.join_schedules([warmup, cosine], [warmup_steps])

    if name == "resume_ramp_cosine":
        ramp_start = int(meta["resume_ramp_start_step"])
        ramp_steps = int(meta["resume_ramp_steps"])
        ramp_init_lr = float(meta["resume_ramp_init_lr"])
        if ramp_steps < 1:
            return optim.cosine_decay(max_lr, decay_steps, end=min_lr)

        cosine_steps = max(1, decay_steps - ramp_start - ramp_steps)

        def schedule(step):
            ramp_position = mx.minimum(mx.maximum(step - ramp_start, 0), ramp_steps)
            ramp_lr = ramp_init_lr + ramp_position * ((max_lr - ramp_init_lr) / ramp_steps)
            cosine_position = mx.minimum(mx.maximum(step - ramp_start - ramp_steps, 0), cosine_steps)
            cosine_lr = min_lr + 0.5 * (max_lr - min_lr) * (
                1 + mx.cos(mx.pi * cosine_position / cosine_steps)
            )
            return mx.where(step < ramp_start + ramp_steps, ramp_lr, cosine_lr)

        return schedule

    raise AssertionError(f"Unhandled lr schedule: {name}")


def current_learning_rate(optimizer) -> float:
    lr = optimizer.learning_rate
    mx.eval(lr)
    return float(lr)


def configured_learning_rate_at_step(cfg: dict[str, Any], step: int) -> mx.array:
    lr = build_learning_rate(cfg)
    if callable(lr):
        value = lr(mx.array(step, mx.uint64))
    else:
        value = mx.array(float(lr), mx.float32)
    mx.eval(value)
    return value


def override_loaded_optimizer_learning_rate(optimizer, cfg: dict[str, Any], step: int) -> bool:
    if "learning_rate" not in optimizer.state:
        return False
    optimizer.state["learning_rate"] = configured_learning_rate_at_step(cfg, step)
    mx.eval(optimizer.state["learning_rate"])
    return True


def canonical_optimizer_name(cfg: dict[str, Any]) -> str:
    name = str(cfg.get("optimizer", "adamw")).strip().lower().replace("_", "-")
    aliases = {
        "adamw": "adamw",
        "adam-w": "adamw",
        "adafactor": "adafactor",
        "lion": "lion",
    }
    if name not in aliases:
        raise ValueError(f"Unsupported optimizer: {cfg.get('optimizer')!r}. Use adamw, adafactor, or lion.")
    return aliases[name]


def build_optimizer(cfg: dict[str, Any]):
    name = canonical_optimizer_name(cfg)
    learning_rate = build_learning_rate(cfg)
    weight_decay = float(cfg.get("weight_decay", 0.1))

    if name == "adamw":
        betas = pair_from_config(cfg.get("adamw_betas", cfg.get("optimizer_betas")), (0.9, 0.95))
        eps = float(cfg.get("adamw_eps", cfg.get("optimizer_eps", 1e-6)))
        bias_correction = bool(cfg.get("adamw_bias_correction", cfg.get("optimizer_bias_correction", False)))
        return optim.AdamW(
            learning_rate=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            bias_correction=bias_correction,
        )

    if name == "adafactor":
        eps = pair_from_config(cfg.get("adafactor_eps"), (1e-30, 1e-3))
        beta_1 = cfg.get("adafactor_beta_1", cfg.get("adafactor_beta1"))
        return optim.Adafactor(
            learning_rate=learning_rate,
            eps=(eps[0], eps[1]),
            clip_threshold=float(cfg.get("adafactor_clip_threshold", 1.0)),
            decay_rate=float(cfg.get("adafactor_decay_rate", -0.8)),
            beta_1=None if beta_1 is None else float(beta_1),
            weight_decay=weight_decay,
            scale_parameter=bool(cfg.get("adafactor_scale_parameter", True)),
            relative_step=bool(cfg.get("adafactor_relative_step", False)),
            warmup_init=bool(cfg.get("adafactor_warmup_init", False)),
        )

    if name == "lion":
        betas = pair_from_config(cfg.get("lion_betas", cfg.get("optimizer_betas")), (0.9, 0.99))
        return optim.Lion(learning_rate=learning_rate, betas=betas, weight_decay=weight_decay)

    raise AssertionError(f"Unhandled optimizer: {name}")


class BatchSampler:
    def __init__(
        self,
        train_path: Path | list[Path],
        val_path: Path | list[Path],
        block_size: int,
        batch_size: int,
        seed: int,
    ):
        # Use mmap so larger long-training token files do not get copied into RAM.
        # The data preparation scripts write int32 arrays; if an older file has a
        # different dtype we still fail loudly instead of silently duplicating it.
        self.train_arrays = self._load_arrays(train_path)
        self.val_arrays = self._load_arrays(val_path)
        self.block_size = block_size
        self.batch_size = batch_size
        self.seed = int(seed)
        self.train_index = self._build_block_index(self.train_arrays, block_size)
        self.val_index = self._build_block_index(self.val_arrays, block_size)
        self.split_states = {
            "train": self._new_split_state("train", self.seed),
            "val": self._new_split_state("val", self.seed + 1_000_003),
        }

    @staticmethod
    def _load_arrays(path_or_paths: Path | list[Path]) -> list[np.ndarray]:
        paths = path_or_paths if isinstance(path_or_paths, list) else [path_or_paths]
        arrays = []
        for path in paths:
            arr = np.load(path, mmap_mode="r")
            if arr.dtype != np.int32:
                raise TypeError(
                    f"Expected int32 token array at {path}, got {arr.dtype}. "
                    "Re-encode the dataset with the streaming encoder."
                )
            arrays.append(arr)
        return arrays

    @staticmethod
    def _build_block_index(arrays: list[np.ndarray], block_size: int) -> dict[str, Any]:
        # A valid block start must have block_size input tokens plus one target
        # token available. Non-overlap starts are 0, block_size, 2*block_size...
        counts = np.asarray([max(0, (len(arr) - 1) // block_size) for arr in arrays], dtype=np.int64)
        if int(counts.sum()) <= 0:
            raise ValueError("Token shards are too small for the configured block_size")
        offsets = np.concatenate([np.asarray([0], dtype=np.int64), np.cumsum(counts)])
        return {
            "counts": counts,
            "offsets": offsets,
            "total_blocks": int(counts.sum()),
        }

    def _new_split_state(self, split: str, permutation_seed: int) -> dict[str, Any]:
        index = self.train_index if split == "train" else self.val_index
        state = {
            "epoch": 0,
            "cursor": 0,
            "permutation_seed": int(permutation_seed),
            "permutation": None,
        }
        state["permutation"] = self._make_permutation(index["total_blocks"], state["permutation_seed"], state["epoch"])
        return state

    @staticmethod
    def _make_permutation(total_blocks: int, permutation_seed: int, epoch: int) -> np.ndarray:
        rng = np.random.default_rng(int(permutation_seed) + int(epoch))
        return rng.permutation(total_blocks).astype(np.int64, copy=False)

    def _state_for_split(self, split: str) -> tuple[list[np.ndarray], dict[str, Any], dict[str, Any]]:
        if split == "train":
            return self.train_arrays, self.train_index, self.split_states["train"]
        if split == "val":
            return self.val_arrays, self.val_index, self.split_states["val"]
        raise ValueError(f"Unsupported split: {split!r}")

    def _take_block_ids(self, split: str, count: int) -> np.ndarray:
        _, index, state = self._state_for_split(split)
        ids = []
        remaining = int(count)
        while remaining > 0:
            total_blocks = int(index["total_blocks"])
            cursor = int(state["cursor"])
            if cursor >= total_blocks:
                state["epoch"] = int(state["epoch"]) + 1
                state["cursor"] = 0
                state["permutation"] = self._make_permutation(
                    total_blocks,
                    int(state["permutation_seed"]),
                    int(state["epoch"]),
                )
                cursor = 0
            take = min(remaining, total_blocks - cursor)
            ids.append(state["permutation"][cursor : cursor + take])
            state["cursor"] = cursor + take
            remaining -= take
        return np.concatenate(ids)

    def _materialize_blocks(self, split: str, block_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arrays, index, _ = self._state_for_split(split)
        offsets = index["offsets"]
        shard_ids = np.searchsorted(offsets, block_ids, side="right") - 1
        xs = []
        ys = []
        for block_id, shard_id in zip(block_ids, shard_ids):
            local_block = int(block_id - offsets[int(shard_id)])
            start = local_block * self.block_size
            data = arrays[int(shard_id)]
            xs.append(data[start : start + self.block_size])
            ys.append(data[start + 1 : start + self.block_size + 1])
        return np.stack(xs), np.stack(ys)

    def get_batch(self, split: str):
        block_ids = self._take_block_ids(split, self.batch_size)
        x, y = self._materialize_blocks(split, block_ids)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "sampler_version": "non_overlap_blocks_v1",
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "splits": {
                split: {
                    "epoch": int(state["epoch"]),
                    "cursor": int(state["cursor"]),
                    "permutation_seed": int(state["permutation_seed"]),
                    "total_blocks": int((self.train_index if split == "train" else self.val_index)["total_blocks"]),
                }
                for split, state in self.split_states.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("sampler_version") != "non_overlap_blocks_v1":
            # Legacy checkpoints stored only a NumPy RNG state for random
            # sampling with replacement. There is no exact cursor equivalent,
            # so a resumed run starts a deterministic non-overlap epoch from
            # this sampler's configured seed.
            return
        for split in ["train", "val"]:
            split_payload = state.get("splits", {}).get(split, {})
            index = self.train_index if split == "train" else self.val_index
            total_blocks = int(index["total_blocks"])
            permutation_seed = int(split_payload.get("permutation_seed", self.split_states[split]["permutation_seed"]))
            epoch = int(split_payload.get("epoch", 0))
            cursor = int(split_payload.get("cursor", 0))
            self.split_states[split] = {
                "epoch": max(0, epoch),
                "cursor": min(max(0, cursor), total_blocks),
                "permutation_seed": permutation_seed,
                "permutation": self._make_permutation(total_blocks, permutation_seed, max(0, epoch)),
            }


class HFTokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


class TokenizersWrapper:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def load_tokenizer(metadata: dict[str, Any]):
    tokenizer_type = metadata.get("tokenizer_type", "qwen")
    if tokenizer_type == "qwen":
        tokenizer_name = metadata["tokenizer_name"]
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            cache_dir=str(CURRENT_DIR / "data/cache/transformers"),
        )
        return HFTokenizerWrapper(tokenizer)
    if tokenizer_type in {"lab_bpe", "byte_bpe"}:
        tokenizer_path = Path(metadata["tokenizer_path"])
        return TokenizersWrapper(Tokenizer.from_file(str(tokenizer_path)))
    raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")


def lm_loss(model, idx, targets):
    logits = model(idx).astype(mx.float32)
    b, t, v = logits.shape
    return nn.losses.cross_entropy(logits.reshape(b * t, v), targets.reshape(b * t), reduction="mean")


def estimate_loss(model, sampler: BatchSampler, eval_iters: int):
    out = {}
    for split in ["train", "val"]:
        eval_sampler_state = sampler.state_dict()
        losses = []
        for _ in range(eval_iters):
            x, y = sampler.get_batch(split)
            loss = lm_loss(model, x, y)
            mx.eval(loss)
            losses.append(float(loss))
        out[split] = sum(losses) / len(losses)
        # Evaluation should be comparable and side-effect-free. If train eval
        # advances the cursor, training silently skips blocks; if val eval
        # advances the cursor, each validation point measures a different
        # slice and best_val becomes noisy.
        sampler.load_state_dict(eval_sampler_state)
    return out


def sample_next(logits, temperature: float, top_k: int):
    logits = logits.astype(mx.float32)
    if temperature <= 0:
        return int(mx.argmax(logits))
    logits = logits / temperature
    if top_k and top_k > 0:
        values = mx.topk(logits, k=top_k)
        threshold = values[-1]
        logits = mx.where(logits < threshold, mx.full(logits.shape, -1e9), logits)
    probs = mx.softmax(logits, axis=-1)
    next_id = mx.random.categorical(mx.log(probs))
    mx.eval(next_id)
    return int(next_id)


def apply_stop_strings(text: str, prompt: str, stop_strings: list[str]) -> tuple[str, bool]:
    if not stop_strings:
        return text, False

    generated_part = text[len(prompt) :] if text.startswith(prompt) else text
    stop_positions = [generated_part.find(stop) for stop in stop_strings if stop and generated_part.find(stop) >= 0]
    if not stop_positions:
        return text, False

    cut = min(stop_positions)
    return prompt + generated_part[:cut], True


def generate(model, tokenizer, prompt: str, cfg: dict, max_new_tokens: int):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    block_size = int(cfg["block_size"])
    stop_strings = list(cfg.get("sample_stop_strings", []))
    for _ in range(max_new_tokens):
        context = ids[-block_size:]
        idx = mx.array([context], dtype=mx.int32)
        logits = model(idx)[0, -1]
        mx.eval(logits)
        ids.append(sample_next(logits, float(cfg.get("temperature", 0.8)), int(cfg.get("top_k", 50))))
        text = tokenizer.decode(ids, skip_special_tokens=True)
        stopped_text, should_stop = apply_stop_strings(text, prompt, stop_strings)
        if should_stop:
            return stopped_text
    return tokenizer.decode(ids, skip_special_tokens=True)


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (CURRENT_DIR.parent / value).resolve()


def companion_checkpoint_paths(model_path: Path) -> dict[str, Path]:
    name = model_path.name
    if not name.endswith("_model.safetensors"):
        stem = model_path.stem
    else:
        stem = name[: -len("_model.safetensors")]
    return {
        "model": model_path,
        "optimizer": model_path.with_name(f"{stem}_optimizer.safetensors"),
        "state": model_path.with_name(f"{stem}_state.json"),
        "meta": model_path.with_name(f"{stem}_meta.json"),
    }


def load_checkpoint_bundle(
    *,
    model,
    optimizer,
    sampler: BatchSampler,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    resume_checkpoint_path = cfg.get("resume_checkpoint_path")
    if not resume_checkpoint_path:
        return {"resumed": False, "previous_steps": int(cfg.get("resume_previous_steps", 0)), "best_val_loss": float("inf")}

    model_path = resolve_path(resume_checkpoint_path)
    paths = companion_checkpoint_paths(model_path)
    print(f"Loading model weights: {paths['model']}", flush=True)
    model.load_weights(str(paths["model"]))

    state_payload = load_json(paths["state"]) if paths["state"].exists() else None
    meta_payload = load_json(paths["meta"]) if paths["meta"].exists() else None
    payload = state_payload or meta_payload or {}

    optimizer_loaded = False
    if paths["optimizer"].exists():
        optimizer.state = tree_unflatten(list(mx.load(str(paths["optimizer"])).items()))
        mx.eval(optimizer.state)
        optimizer_loaded = True
        print(f"Loaded optimizer state: {paths['optimizer']}", flush=True)
    else:
        print("No optimizer state found for checkpoint; continuing with fresh optimizer state.", flush=True)

    if state_payload:
        sampler_payload = state_payload.get("sampler_state") or state_payload.get("sampler_rng_state")
        if sampler_payload:
            sampler.load_state_dict(sampler_payload)

    previous_steps = int(
        cfg.get(
            "resume_previous_steps",
            payload.get("global_step", int(payload.get("step", -1)) + 1),
        )
    )
    previous_tokens_seen = int(
        cfg.get(
            "resume_previous_tokens_seen",
            payload.get("total_tokens_seen", 0),
        )
    )
    best_val_loss = float(
        cfg.get(
            "resume_best_val_loss",
            payload.get("best_val_loss", payload.get("val_loss", float("inf"))),
        )
    )
    return {
        "resumed": True,
        "model_path": str(paths["model"]),
        "optimizer_path": str(paths["optimizer"]) if paths["optimizer"].exists() else None,
        "state_path": str(paths["state"]) if paths["state"].exists() else None,
        "meta_path": str(paths["meta"]) if paths["meta"].exists() else None,
        "optimizer_loaded": optimizer_loaded,
        "previous_steps": previous_steps,
        "previous_tokens_seen": previous_tokens_seen,
        "best_val_loss": best_val_loss,
    }


def save_checkpoint(
    model,
    optimizer,
    sampler: BatchSampler,
    run_dir: Path,
    tag: str,
    step: int,
    metrics: dict[str, Any],
    *,
    best_val_loss: float,
    previous_steps: int,
    previous_tokens_seen: int,
    run_steps: int,
    tokens_per_step: int,
    stop_reason: str = "running",
) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_path = ckpt_dir / f"{tag}_model.safetensors"
    optimizer_path = ckpt_dir / f"{tag}_optimizer.safetensors"
    state_path = ckpt_dir / f"{tag}_state.json"
    model.save_weights(str(weights_path))
    mx.save_safetensors(str(optimizer_path), dict(tree_flatten(optimizer.state)))
    mx.eval(model.parameters(), optimizer.state)
    state = {
        "tag": tag,
        "step": step,
        "global_step": step + 1,
        "previous_steps": previous_steps,
        "previous_tokens_seen": previous_tokens_seen,
        "run_steps": run_steps,
        "tokens_per_step": tokens_per_step,
        "total_tokens_seen": previous_tokens_seen + run_steps * tokens_per_step,
        "run_tokens_seen": run_steps * tokens_per_step,
        "best_val_loss": best_val_loss,
        "stop_reason": stop_reason,
        "weights_path": str(weights_path),
        "optimizer_path": str(optimizer_path),
        "sampler_state": sampler.state_dict(),
        # Backward-compatible field name for older dashboard/report readers.
        "sampler_rng_state": sampler.state_dict(),
        **metrics,
    }
    write_json(state_path, state)
    write_json(ckpt_dir / f"{tag}_meta.json", state)


def plot_loss(history: list[dict[str, Any]], output_path: Path) -> None:
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in history]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, [r["train_loss"] for r in history], marker="o", label="train")
    plt.plot(steps, [r["val_loss"] for r in history], marker="o", label="val")
    plt.xlabel("step")
    plt.ylabel("cross entropy")
    plt.title("Qwen-like Maxout Training Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_run_dir(run_name: str) -> Path:
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{run_name}"
    run_dir = CURRENT_DIR / "outputs" / "runs" / run_id
    for sub in ["samples", "checkpoints", "evals", "benchmarks"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def run_dir_from_checkpoint_path(raw_path: Any) -> Path | None:
    if not raw_path:
        return None
    checkpoint_path = resolve_path(str(raw_path))
    if checkpoint_path.name.endswith(".safetensors") and checkpoint_path.parent.name == "checkpoints":
        candidate = checkpoint_path.parent.parent
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    if checkpoint_path.name.endswith(".safetensors"):
        companions = companion_checkpoint_paths(checkpoint_path)
        for companion in [companions["state"], companions["meta"]]:
            if not companion.exists():
                continue
            try:
                payload = load_json(companion)
            except Exception:
                continue
            for key in ["weights_path", "model_path", "checkpoint_path", "optimizer_path"]:
                source_path = payload.get(key)
                source_run_dir = run_dir_from_checkpoint_path(source_path)
                if source_run_dir:
                    return source_run_dir
    return None


def resume_history_run_dirs(cfg: dict[str, Any], current_run_dir: Path, max_depth: int = 16) -> list[Path]:
    source_dirs: list[Path] = []
    seen = {current_run_dir.resolve()}
    checkpoint = cfg.get("resume_checkpoint_path")

    for _ in range(max_depth):
        source_dir = run_dir_from_checkpoint_path(checkpoint)
        if not source_dir or source_dir in seen:
            break
        source_dirs.append(source_dir)
        seen.add(source_dir)
        source_config_path = source_dir / "config.json"
        if not source_config_path.exists():
            break
        try:
            source_config = load_json(source_config_path)
        except Exception:
            break
        checkpoint = source_config.get("resume_checkpoint_path")

    return list(reversed(source_dirs))


def import_resume_training_history(run_dir: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(cfg.get("include_resume_history", True)):
        return []
    log_path = run_dir / "training_log.jsonl"
    if log_path.exists() and log_path.stat().st_size > 0:
        return read_jsonl(log_path)

    rows_by_key: dict[tuple[Any, Any], dict[str, Any]] = {}
    imported_sources: list[dict[str, Any]] = []
    for source_dir in resume_history_run_dirs(cfg, run_dir):
        source_rows = read_jsonl(source_dir / "training_log.jsonl")
        if not source_rows:
            continue
        imported_sources.append({"run_id": source_dir.name, "rows": len(source_rows)})
        for row in source_rows:
            next_row = {**row}
            next_row.setdefault("history_source_run_id", source_dir.name)
            key = (next_row.get("step"), next_row.get("tokens_seen"))
            rows_by_key[key] = next_row

    imported_rows = sorted(
        rows_by_key.values(),
        key=lambda row: (
            int(row.get("step", 0)) if isinstance(row.get("step"), int) else 0,
            int(row.get("tokens_seen", 0)) if isinstance(row.get("tokens_seen"), int) else 0,
        ),
    )
    if imported_rows:
        for row in imported_rows:
            append_jsonl(log_path, row)
    write_json(
        run_dir / "resume_history_manifest.json",
        {
            "enabled": True,
            "imported_rows": len(imported_rows),
            "sources": imported_sources,
        },
    )
    return imported_rows


def deadline_from_local_time(value: str) -> float:
    hour_text, minute_text = value.split(":", 1)
    now = datetime.now()
    deadline = now.replace(hour=int(hour_text), minute=int(minute_text), second=0, microsecond=0)
    if deadline <= now:
        deadline += timedelta(days=1)
    return deadline.timestamp()


def train(config_path: Path) -> Path:
    cfg = load_json(config_path)
    np.random.seed(int(cfg.get("seed", 2030)))
    mx.random.seed(int(cfg.get("seed", 2030)))

    if cfg.get("memory_limit_gb"):
        mx.set_memory_limit(int(float(cfg["memory_limit_gb"]) * 1024**3))

    metadata_path = Path(cfg.get("metadata_path", CURRENT_DIR / "data" / "metadata" / "qwen_token_data_metadata.json"))
    metadata = load_json(metadata_path)
    data_dir = CURRENT_DIR / "data" / "processed"
    if metadata.get("train_shards") and metadata.get("val_shards"):
        train_path = [Path(item["path"]) for item in metadata["train_shards"]]
        val_path = [Path(item["path"]) for item in metadata["val_shards"]]
    else:
        train_path = Path(cfg.get("train_tokens_path", metadata.get("train_tokens_path", data_dir / "train_tokens.npy")))
        val_path = Path(cfg.get("val_tokens_path", metadata.get("val_tokens_path", data_dir / "val_tokens.npy")))
    vocab_size = int(metadata["vocab_size"])
    tokenizer = load_tokenizer(metadata)

    model_cfg = QwenLikeConfig(
        vocab_size=vocab_size,
        block_size=int(cfg["block_size"]),
        n_embd=int(cfg["n_embd"]),
        num_layers=int(cfg["num_layers"]),
        num_q_heads=int(cfg["num_q_heads"]),
        num_kv_heads=int(cfg["num_kv_heads"]),
        ffn_multiplier=float(cfg.get("ffn_multiplier", 3.0)),
        rope_base=float(cfg.get("rope_base", 1_000_000.0)),
        qk_norm=bool(cfg.get("qk_norm", True)),
        weight_tying=bool(cfg.get("weight_tying", True)),
        use_bias=bool(cfg.get("use_bias", False)),
        activation_checkpointing=bool(cfg.get("activation_checkpointing", False)),
        rope_impl=str(cfg.get("rope_impl", "manual")),
        rms_norm_impl=str(cfg.get("rms_norm_impl", "manual")),
    )
    model = QwenLikeDenseLM(model_cfg)
    dtype_name = cfg.get("dtype", "bfloat16")
    if dtype_name != "float32":
        model.set_dtype(dtype_from_name(dtype_name))

    optimizer_name = canonical_optimizer_name(cfg)
    lr_schedule_info = lr_schedule_metadata(cfg)
    optimizer = build_optimizer(cfg)
    raw_value_and_grad = nn.value_and_grad(model, lm_loss)
    compile_value_and_grad = bool(cfg.get("compile_value_and_grad", False))
    gradient_accumulation_steps = int(cfg.get("gradient_accumulation_steps", cfg.get("grad_accum_steps", 1)))
    grad_clip_norm = float(cfg.get("grad_clip_norm", cfg.get("gradient_clip_norm", 0.0)) or 0.0)
    monitor_grad_norm = bool(cfg.get("monitor_grad_norm", True))
    loss_spike_window = int(cfg.get("loss_spike_window", 50))
    loss_spike_threshold = float(cfg.get("loss_spike_threshold", 1.35))
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    value_and_grad = raw_value_and_grad
    micro_batch_size = int(cfg["batch_size"])
    effective_batch_size = micro_batch_size * gradient_accumulation_steps
    sampler = BatchSampler(
        train_path,
        val_path,
        block_size=int(cfg["block_size"]),
        batch_size=micro_batch_size,
        seed=int(cfg.get("seed", 2030)),
    )
    resume_info = load_checkpoint_bundle(model=model, optimizer=optimizer, sampler=sampler, cfg=cfg)
    if resume_info.get("resumed") and resume_info.get("optimizer_loaded"):
        previous_step_count = int(resume_info.get("previous_steps", 0))
        if override_loaded_optimizer_learning_rate(optimizer, cfg, previous_step_count):
            print(
                "Overrode loaded optimizer learning_rate from current config: "
                f"{float(optimizer.state['learning_rate']):.6g}",
                flush=True,
            )
    if resume_info.get("resumed") and not resume_info.get("optimizer_loaded"):
        previous_step_count = int(resume_info.get("previous_steps", 0))
        if previous_step_count > 0:
            # If only model weights are available, keep the LR schedule aligned
            # with global training progress instead of restarting warmup at 0.
            optimizer.state["step"] = mx.array(previous_step_count, mx.uint64)
    mx.eval(model.parameters(), optimizer.state)

    compiled_train_step = None
    if compile_value_and_grad:
        # Compile the whole mutable train step, not a closure that only returns
        # gradients. Capturing model.state and optimizer.state as both inputs and
        # outputs lets MLX read the latest parameters and write back updates.
        # The old `mx.compile(lambda x, y: value_and_grad(model, x, y))` form can
        # freeze model parameters inside the compiled graph and produce stale
        # train-loss telemetry.
        if gradient_accumulation_steps == 1:
            def train_step_compiled(x, y):
                loss, grads = raw_value_and_grad(model, x, y)
                if grad_clip_norm > 0:
                    grads, grad_norm = optim.clip_grad_norm(grads, grad_clip_norm)
                elif monitor_grad_norm:
                    grad_norm = tree_l2_norm(grads)
                else:
                    grad_norm = mx.array(float("nan"), dtype=mx.float32)
                optimizer.update(model, grads)
                return loss, grad_norm
        else:
            def train_step_compiled(x, y):
                accumulated_grads = None
                total_loss = mx.array(0.0, dtype=mx.float32)
                for micro_step in range(gradient_accumulation_steps):
                    loss, grads = raw_value_and_grad(model, x[micro_step], y[micro_step])
                    total_loss = total_loss + loss / gradient_accumulation_steps
                    scaled_grads = tree_map(lambda g: g / gradient_accumulation_steps, grads)
                    accumulated_grads = (
                        scaled_grads
                        if accumulated_grads is None
                        else tree_map(lambda total, grad: total + grad, accumulated_grads, scaled_grads)
                    )
                if grad_clip_norm > 0:
                    accumulated_grads, grad_norm = optim.clip_grad_norm(accumulated_grads, grad_clip_norm)
                elif monitor_grad_norm:
                    grad_norm = tree_l2_norm(accumulated_grads)
                else:
                    grad_norm = mx.array(float("nan"), dtype=mx.float32)
                optimizer.update(model, accumulated_grads)
                return total_loss, grad_norm

        compiled_train_step = mx.compile(
            train_step_compiled,
            inputs=[model.state, optimizer.state],
            outputs=[model.state, optimizer.state],
        )

    shape_sampler_state = sampler.state_dict()
    try:
        x0, _ = sampler.get_batch("train")
        shape_info = model.inspect_shapes(x0)
    finally:
        sampler.load_state_dict(shape_sampler_state)
    param_count = count_parameters(model.parameters())

    run_dir = make_run_dir(cfg["run_name"])
    cfg = {
        **cfg,
        "model_type": "qwen_like_dense_maxout",
        "tokenizer_type": metadata.get("tokenizer_type", "qwen"),
        "tokenizer_name": metadata.get("tokenizer_name"),
        "tokenizer_path": metadata.get("tokenizer_path"),
        "metadata_path": str(metadata_path),
        "vocab_size": vocab_size,
        "parameter_count": param_count,
        "shape_info": shape_info,
        "resume_info": resume_info,
        "optimizer": optimizer_name,
        "lr_schedule_config": lr_schedule_info,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "grad_clip_norm": grad_clip_norm,
        "monitor_grad_norm": monitor_grad_norm,
        "loss_spike_window": loss_spike_window,
        "loss_spike_threshold": loss_spike_threshold,
        "activation_checkpointing": model_cfg.activation_checkpointing,
        "compile_value_and_grad": compile_value_and_grad,
        "rope_impl": model_cfg.rope_impl,
        "rms_norm_impl": model_cfg.rms_norm_impl,
    }
    write_json(run_dir / "config.json", cfg)
    write_json(run_dir / "original_config.json", load_json(config_path))
    imported_history = import_resume_training_history(run_dir, cfg)

    max_iters = int(cfg["max_iters"])
    deadline_ts = None
    stop_at_local_time = cfg.get("stop_at_local_time")
    if stop_at_local_time:
        deadline_ts = deadline_from_local_time(str(stop_at_local_time))
    eval_interval = int(cfg.get("eval_interval", 50))
    eval_iters = int(cfg.get("eval_iters", 4))
    heartbeat_interval = int(cfg.get("heartbeat_interval", 10))
    sample_interval = int(cfg.get("sample_interval", eval_interval))
    checkpoint_interval = int(cfg.get("checkpoint_interval", eval_interval))
    save_checkpoints_enabled = bool(cfg.get("save_checkpoints", True))
    generate_samples_enabled = bool(cfg.get("generate_samples", True))
    tokens_per_step = micro_batch_size * int(cfg["block_size"]) * gradient_accumulation_steps
    history = list(imported_history)
    previous_steps = int(resume_info.get("previous_steps", 0))
    previous_tokens_seen = int(resume_info.get("previous_tokens_seen", previous_steps * tokens_per_step))
    best_val = float(resume_info.get("best_val_loss", float("inf")))
    start = time.perf_counter()
    last_step_start = start
    last_step_time_ms = None
    last_completed_step = previous_steps - 1
    last_eval_step = None
    last_eval_train_loss = None
    last_eval_val_loss = None
    recent_train_losses: list[float] = [
        float(row["train_loss"])
        for row in history[-max(loss_spike_window * 2, 100) :]
        if isinstance(row.get("train_loss"), (int, float))
    ]
    stop_reason = "max_iters"

    mx.reset_peak_memory()
    if hasattr(mx, "metal"):
        mx.metal.reset_peak_memory()

    for local_step in range(max_iters):
        step = previous_steps + local_step
        step_start = time.perf_counter()
        grad_norm_value = None
        grad_clip_scale = None
        update_norm_estimate = None
        loss_spike = False
        loss_spike_baseline = None
        if compiled_train_step is not None:
            if gradient_accumulation_steps == 1:
                x, y = sampler.get_batch("train")
            else:
                xs = []
                ys = []
                for _ in range(gradient_accumulation_steps):
                    x_micro, y_micro = sampler.get_batch("train")
                    xs.append(x_micro)
                    ys.append(y_micro)
                x = mx.stack(xs, axis=0)
                y = mx.stack(ys, axis=0)
            loss, grad_norm = compiled_train_step(x, y)
            mx.eval(model.parameters(), optimizer.state, loss, grad_norm)
            loss_value = float(loss)
            grad_norm_value = float(grad_norm) if math.isfinite(float(grad_norm)) else None
            if not math.isfinite(loss_value):
                write_json(
                    run_dir / "status.json",
                    {
                        "run_id": run_dir.name,
                        "state": "failed",
                        "step": step,
                        "micro_step": 0,
                        "reason": "non-finite loss",
                    },
                )
                raise RuntimeError(f"Non-finite loss at step {step}: {loss_value}")
            train_loss_value = loss_value
        else:
            accumulated_grads = None
            micro_losses = []
            for micro_step in range(gradient_accumulation_steps):
                x, y = sampler.get_batch("train")
                loss, grads = value_and_grad(model, x, y)
                scaled_grads = tree_map(lambda g: g / gradient_accumulation_steps, grads)
                accumulated_grads = (
                    scaled_grads
                    if accumulated_grads is None
                    else tree_map(lambda total, grad: total + grad, accumulated_grads, scaled_grads)
                )
                mx.eval(loss, accumulated_grads)
                loss_value = float(loss)
                if not math.isfinite(loss_value):
                    write_json(
                        run_dir / "status.json",
                        {
                            "run_id": run_dir.name,
                            "state": "failed",
                            "step": step,
                            "micro_step": micro_step,
                            "reason": "non-finite loss",
                        },
                    )
                    raise RuntimeError(f"Non-finite loss at step {step} micro_step {micro_step}: {loss_value}")
                micro_losses.append(loss_value)

            if grad_clip_norm > 0:
                accumulated_grads, grad_norm = optim.clip_grad_norm(accumulated_grads, grad_clip_norm)
            elif monitor_grad_norm:
                grad_norm = tree_l2_norm(accumulated_grads)
            else:
                grad_norm = mx.array(float("nan"), dtype=mx.float32)
            optimizer.update(model, accumulated_grads)
            mx.eval(model.parameters(), optimizer.state, grad_norm)
            grad_norm_value = float(grad_norm) if math.isfinite(float(grad_norm)) else None
            train_loss_value = sum(micro_losses) / max(len(micro_losses), 1)
        last_step_time_ms = (time.perf_counter() - step_start) * 1000
        learning_rate_value = current_learning_rate(optimizer)
        if grad_norm_value is not None:
            grad_clip_scale = (
                min(grad_clip_norm / (grad_norm_value + 1e-6), 1.0)
                if grad_clip_norm > 0
                else 1.0
            )
            update_norm_estimate = learning_rate_value * min(grad_norm_value, grad_clip_norm if grad_clip_norm > 0 else grad_norm_value)
        if len(recent_train_losses) >= max(5, min(loss_spike_window, 10)):
            window = recent_train_losses[-loss_spike_window:]
            loss_spike_baseline = float(np.median(window))
            loss_spike = bool(loss_spike_baseline > 0 and train_loss_value > loss_spike_baseline * loss_spike_threshold)
        recent_train_losses.append(train_loss_value)
        if len(recent_train_losses) > max(loss_spike_window * 2, 100):
            recent_train_losses = recent_train_losses[-max(loss_spike_window * 2, 100):]

        should_eval = local_step % eval_interval == 0 or local_step == max_iters - 1
        if should_eval:
            losses = estimate_loss(model, sampler, eval_iters)
            last_eval_step = step
            last_eval_train_loss = losses["train"]
            last_eval_val_loss = losses["val"]
            elapsed = time.perf_counter() - start
            run_steps = local_step + 1
            run_tokens_seen = run_steps * tokens_per_step
            tokens_seen = previous_tokens_seen + run_tokens_seen
            tokens_per_second = run_tokens_seen / max(elapsed, 1e-9)
            remaining_steps = max_iters - local_step - 1
            eta_sec = remaining_steps * (elapsed / max(run_steps, 1))
            row = {
                "step": step,
                "local_step": local_step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "tokens_seen": tokens_seen,
                "run_tokens_seen": run_tokens_seen,
                "tokens_per_second": tokens_per_second,
                "learning_rate": learning_rate_value,
                "max_learning_rate": lr_schedule_info["max_learning_rate"],
                "min_learning_rate": lr_schedule_info["min_learning_rate"],
                "lr_schedule": lr_schedule_info["lr_schedule"],
                "grad_norm": grad_norm_value,
                "grad_clip_norm": grad_clip_norm if grad_clip_norm > 0 else None,
                "grad_clip_scale": grad_clip_scale,
                "update_norm_estimate": update_norm_estimate,
                "loss_spike": loss_spike,
                "loss_spike_baseline": loss_spike_baseline,
                "optimizer": optimizer_name,
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "activation_checkpointing": model_cfg.activation_checkpointing,
                "compile_value_and_grad": compile_value_and_grad,
                "rope_impl": model_cfg.rope_impl,
                "rms_norm_impl": model_cfg.rms_norm_impl,
                "elapsed_sec": elapsed,
            }
            history.append(row)
            append_jsonl(run_dir / "training_log.jsonl", row)
            if losses["val"] < best_val:
                best_val = losses["val"]
                if save_checkpoints_enabled:
                    save_checkpoint(
                        model,
                        optimizer,
                        sampler,
                        run_dir,
                        "best_val",
                        step,
                        {
                            "val_loss": best_val,
                            "learning_rate": learning_rate_value,
                            "lr_schedule": lr_schedule_info["lr_schedule"],
                        },
                        best_val_loss=best_val,
                        previous_steps=previous_steps,
                        previous_tokens_seen=previous_tokens_seen,
                        run_steps=run_steps,
                        tokens_per_step=tokens_per_step,
                    )

            status = {
                "run_id": run_dir.name,
                "state": "running" if local_step < max_iters - 1 else "completed",
                "step": step,
                "local_step": local_step,
                "max_iters": max_iters,
                "previous_steps": previous_steps,
                "actual_steps": run_steps,
                "progress_percent": 100.0 * run_steps / max_iters,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "last_eval_step": last_eval_step,
                "last_eval_train_loss": last_eval_train_loss,
                "last_eval_val_loss": last_eval_val_loss,
                "last_train_loss": train_loss_value,
                "status_source": "eval",
                "best_val_loss": best_val,
                "tokens_seen": tokens_seen,
                "run_tokens_seen": run_tokens_seen,
                "tokens_per_second": tokens_per_second,
                "elapsed_sec": elapsed,
                "eta_sec": eta_sec,
                "learning_rate": learning_rate_value,
                "max_learning_rate": lr_schedule_info["max_learning_rate"],
                "min_learning_rate": lr_schedule_info["min_learning_rate"],
                "lr_schedule": lr_schedule_info["lr_schedule"],
                "grad_norm": grad_norm_value,
                "grad_clip_norm": grad_clip_norm if grad_clip_norm > 0 else None,
                "grad_clip_scale": grad_clip_scale,
                "update_norm_estimate": update_norm_estimate,
                "loss_spike": loss_spike,
                "loss_spike_baseline": loss_spike_baseline,
                "optimizer": optimizer_name,
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "activation_checkpointing": model_cfg.activation_checkpointing,
                "compile_value_and_grad": compile_value_and_grad,
                "rope_impl": model_cfg.rope_impl,
                "rms_norm_impl": model_cfg.rms_norm_impl,
                "performance": telemetry_snapshot(
                    step_time_ms=last_step_time_ms,
                    tokens_per_second=tokens_per_second,
                    eta_sec=eta_sec,
                    progress_percent=100.0 * run_steps / max_iters,
                ),
            }
            write_json(run_dir / "status.json", status)
            peak = status["performance"].get("mlx_peak_memory_gb")
            peak_text = f"{peak:.2f}GB" if isinstance(peak, (int, float)) else "n/a"
            print(
                f"step={step:05d} train={losses['train']:.4f} val={losses['val']:.4f} "
                f"lr={learning_rate_value:.2e} tok/s={tokens_per_second:.1f} peak={peak_text}",
                flush=True,
            )
        elif heartbeat_interval > 0 and (
            local_step % heartbeat_interval == 0 or local_step == max_iters - 1
        ):
            elapsed = time.perf_counter() - start
            run_steps = local_step + 1
            run_tokens_seen = run_steps * tokens_per_step
            tokens_seen = previous_tokens_seen + run_tokens_seen
            tokens_per_second = run_tokens_seen / max(elapsed, 1e-9)
            remaining_steps = max_iters - local_step - 1
            eta_sec = remaining_steps * (elapsed / max(run_steps, 1))
            status = {
                "run_id": run_dir.name,
                "state": "running" if local_step < max_iters - 1 else "completed",
                "step": step,
                "local_step": local_step,
                "max_iters": max_iters,
                "previous_steps": previous_steps,
                "actual_steps": run_steps,
                "progress_percent": 100.0 * run_steps / max_iters,
                "train_loss": train_loss_value,
                "val_loss": last_eval_val_loss,
                "last_eval_step": last_eval_step,
                "last_eval_train_loss": last_eval_train_loss,
                "last_eval_val_loss": last_eval_val_loss,
                "last_train_loss": train_loss_value,
                "status_source": "heartbeat",
                "best_val_loss": best_val if math.isfinite(best_val) else None,
                "tokens_seen": tokens_seen,
                "run_tokens_seen": run_tokens_seen,
                "tokens_per_second": tokens_per_second,
                "elapsed_sec": elapsed,
                "eta_sec": eta_sec,
                "learning_rate": learning_rate_value,
                "max_learning_rate": lr_schedule_info["max_learning_rate"],
                "min_learning_rate": lr_schedule_info["min_learning_rate"],
                "lr_schedule": lr_schedule_info["lr_schedule"],
                "grad_norm": grad_norm_value,
                "grad_clip_norm": grad_clip_norm if grad_clip_norm > 0 else None,
                "grad_clip_scale": grad_clip_scale,
                "update_norm_estimate": update_norm_estimate,
                "loss_spike": loss_spike,
                "loss_spike_baseline": loss_spike_baseline,
                "optimizer": optimizer_name,
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "activation_checkpointing": model_cfg.activation_checkpointing,
                "compile_value_and_grad": compile_value_and_grad,
                "rope_impl": model_cfg.rope_impl,
                "rms_norm_impl": model_cfg.rms_norm_impl,
                "performance": telemetry_snapshot(
                    step_time_ms=last_step_time_ms,
                    tokens_per_second=tokens_per_second,
                    eta_sec=eta_sec,
                    progress_percent=100.0 * run_steps / max_iters,
                ),
            }
            write_json(run_dir / "status.json", status)
            append_jsonl(
                run_dir / "heartbeat_log.jsonl",
                {
                    "step": step,
                    "local_step": local_step,
                    "train_loss": train_loss_value,
                    "last_eval_step": last_eval_step,
                    "last_eval_val_loss": last_eval_val_loss,
                    "tokens_seen": tokens_seen,
                    "run_tokens_seen": run_tokens_seen,
                    "tokens_per_second": tokens_per_second,
                    "elapsed_sec": elapsed,
                    "learning_rate": learning_rate_value,
                    "max_learning_rate": lr_schedule_info["max_learning_rate"],
                    "min_learning_rate": lr_schedule_info["min_learning_rate"],
                    "lr_schedule": lr_schedule_info["lr_schedule"],
                    "grad_norm": grad_norm_value,
                    "grad_clip_norm": grad_clip_norm if grad_clip_norm > 0 else None,
                    "grad_clip_scale": grad_clip_scale,
                    "update_norm_estimate": update_norm_estimate,
                    "loss_spike": loss_spike,
                    "loss_spike_baseline": loss_spike_baseline,
                    "status_source": "heartbeat",
                },
            )
            print(
                f"heartbeat step={step:05d} train={train_loss_value:.4f} "
                f"last_eval_val={last_eval_val_loss if last_eval_val_loss is not None else 'n/a'} "
                f"lr={learning_rate_value:.2e} tok/s={tokens_per_second:.1f}",
                flush=True,
            )

        if generate_samples_enabled and (local_step % sample_interval == 0 or local_step == max_iters - 1):
            text = generate(model, tokenizer, cfg.get("prompt", "人工智能"), cfg, int(cfg.get("max_new_tokens", 120)))
            (run_dir / "samples" / f"sample_step_{step:06d}.txt").write_text(text, encoding="utf-8")

        should_save_periodic_checkpoint = (
            checkpoint_interval > 0
            and local_step > 0
            and (local_step % checkpoint_interval == 0 or local_step == max_iters - 1)
        )
        if save_checkpoints_enabled and should_save_periodic_checkpoint:
            save_checkpoint(
                model,
                optimizer,
                sampler,
                run_dir,
                "latest",
                step,
                {
                    "train_loss": train_loss_value,
                    "learning_rate": learning_rate_value,
                    "lr_schedule": lr_schedule_info["lr_schedule"],
                },
                best_val_loss=best_val,
                previous_steps=previous_steps,
                previous_tokens_seen=previous_tokens_seen,
                run_steps=local_step + 1,
                tokens_per_step=tokens_per_step,
            )

        last_completed_step = step
        if deadline_ts is not None and time.time() >= deadline_ts:
            stop_reason = "time_limit"
            break

    elapsed = time.perf_counter() - start
    final_learning_rate_value = current_learning_rate(optimizer)
    plot_loss(history, run_dir / "loss_curve.png")
    final_text = ""
    if generate_samples_enabled:
        final_text = generate(model, tokenizer, cfg.get("prompt", "人工智能"), cfg, int(cfg.get("max_new_tokens", 160)))
    (run_dir / "final_generated_text.txt").write_text(final_text, encoding="utf-8")
    actual_steps = max(0, last_completed_step - previous_steps + 1)
    if save_checkpoints_enabled:
        save_checkpoint(
            model,
            optimizer,
            sampler,
            run_dir,
            "final",
            last_completed_step,
            {
                "best_val_loss": best_val,
                "stop_reason": stop_reason,
                "learning_rate": final_learning_rate_value,
                "lr_schedule": lr_schedule_info["lr_schedule"],
            },
            best_val_loss=best_val,
            previous_steps=previous_steps,
            previous_tokens_seen=previous_tokens_seen,
            run_steps=actual_steps,
            tokens_per_step=tokens_per_step,
            stop_reason=stop_reason,
        )
    global_steps = previous_steps + actual_steps
    metrics = {
        "run_name": cfg["run_name"],
        "model_type": "qwen_like_dense_maxout",
        "parameter_count": param_count,
        "dtype": dtype_name,
        "optimizer": optimizer_name,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "activation_checkpointing": model_cfg.activation_checkpointing,
        "compile_value_and_grad": compile_value_and_grad,
        "rope_impl": model_cfg.rope_impl,
        "rms_norm_impl": model_cfg.rms_norm_impl,
        "max_iters": max_iters,
        "previous_steps": previous_steps,
        "previous_tokens_seen": previous_tokens_seen,
        "actual_steps": actual_steps,
        "global_steps": global_steps,
        "stop_reason": stop_reason,
        "tokens_seen": previous_tokens_seen + actual_steps * tokens_per_step,
        "run_tokens_seen": actual_steps * tokens_per_step,
        "elapsed_sec": elapsed,
        "tokens_per_second": (actual_steps * tokens_per_step) / max(elapsed, 1e-9),
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "best_val_loss": best_val,
        "learning_rate": final_learning_rate_value,
        "lr_schedule_config": lr_schedule_info,
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "metrics.json", metrics)
    if history:
        latest = history[-1]
        status = {
            "run_id": run_dir.name,
            "state": "completed",
            "stop_reason": stop_reason,
            "step": last_completed_step,
            "previous_steps": previous_steps,
            "previous_tokens_seen": previous_tokens_seen,
            "max_iters": max_iters,
            "actual_steps": actual_steps,
            "global_steps": global_steps,
            "progress_percent": 100.0 * actual_steps / max(max_iters, 1),
            "train_loss": latest.get("train_loss"),
            "val_loss": latest.get("val_loss"),
            "best_val_loss": best_val,
            "tokens_seen": previous_tokens_seen + actual_steps * tokens_per_step,
            "run_tokens_seen": actual_steps * tokens_per_step,
            "tokens_per_second": metrics["tokens_per_second"],
            "elapsed_sec": elapsed,
            "eta_sec": 0.0,
            "learning_rate": final_learning_rate_value,
            "max_learning_rate": lr_schedule_info["max_learning_rate"],
            "min_learning_rate": lr_schedule_info["min_learning_rate"],
            "lr_schedule": lr_schedule_info["lr_schedule"],
            "optimizer": optimizer_name,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "activation_checkpointing": model_cfg.activation_checkpointing,
            "compile_value_and_grad": compile_value_and_grad,
            "rope_impl": model_cfg.rope_impl,
            "rms_norm_impl": model_cfg.rms_norm_impl,
            "performance": telemetry_snapshot(
                step_time_ms=last_step_time_ms,
                tokens_per_second=metrics["tokens_per_second"],
                eta_sec=0.0,
                progress_percent=100.0 * actual_steps / max(max_iters, 1),
            ),
        }
        write_json(run_dir / "status.json", status)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_dir = train(Path(args.config))
    print("Run dir:", run_dir)


if __name__ == "__main__":
    main()
