from __future__ import annotations

import numpy as np


def sample_next_id(logits, temperature: float = 0.8, top_k: int = 0, top_p: float = 1.0) -> int:
    logits_np = np.array(logits, dtype=np.float64)
    logits_np = logits_np / max(float(temperature), 1e-6)
    logits_np = logits_np - np.max(logits_np)
    probs = np.exp(logits_np)
    probs = probs / probs.sum()

    if top_k and 0 < top_k < len(probs):
        keep = np.argpartition(probs, -top_k)[-top_k:]
        filtered = np.zeros_like(probs)
        filtered[keep] = probs[keep]
        probs = filtered / filtered.sum()

    if top_p < 1.0:
        order = np.argsort(probs)[::-1]
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)
        keep_mask = cumulative <= top_p
        keep_mask[0] = True
        keep_ids = order[keep_mask]
        filtered = np.zeros_like(probs)
        filtered[keep_ids] = probs[keep_ids]
        probs = filtered / filtered.sum()

    return int(np.random.choice(len(probs), p=probs))
