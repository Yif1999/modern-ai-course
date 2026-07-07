from __future__ import annotations

from typing import Any

from .base import join_nonempty


def format_dialogue_turns(turns: list[str], speakers: tuple[str, str] = ("甲", "乙")) -> str:
    lines = []
    for i, turn in enumerate(turns):
        text = str(turn).strip()
        if not text:
            continue
        speaker = speakers[i % len(speakers)]
        if text.startswith(("甲：", "乙：", "用户：", "助手：")):
            lines.append(text)
        else:
            lines.append(f"{speaker}：{text}")
    return "\n".join(lines).strip()


def format_qa(
    question: str,
    answer: str,
    *,
    context: str | None = None,
    question_label: str = "问题",
    answer_label: str = "回答",
) -> str:
    question = str(question or "").strip()
    context = str(context or "").strip()
    if context and context not in question:
        question = f"{question}\n补充：{context}"
    return join_nonempty([f"{question_label}：{question}", f"{answer_label}：{answer}"])


def format_instruction(instruction: str, response: str, input_text: str | None = None) -> str:
    prompt = instruction
    if input_text:
        prompt = f"{instruction}\n输入：{input_text}"
    return join_nonempty([f"指令：{prompt}", f"回答：{response}"])


def conversation_values(conversations: Any) -> list[tuple[str, str]]:
    if not isinstance(conversations, list):
        return []
    out = []
    for item in conversations:
        if not isinstance(item, dict):
            continue
        role = str(item.get("from", item.get("role", ""))).strip()
        value = str(item.get("value", item.get("content", ""))).strip()
        if value:
            out.append((role, value))
    return out


def format_conversations(conversations: Any) -> str:
    role_map = {
        "human": "用户",
        "user": "用户",
        "assistant": "助手",
        "gpt": "助手",
        "bot": "助手",
    }
    lines = []
    for role, value in conversation_values(conversations):
        label = role_map.get(role.lower(), role or "文本")
        lines.append(f"{label}：{value}")
    return "\n".join(lines).strip()
