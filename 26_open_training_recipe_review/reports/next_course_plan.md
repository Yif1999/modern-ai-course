# 后续 27-35 课程建议

原则：

- 中文优先。
- MLX 优先。
- M1 Pro / 32GB 可实践优先。
- 继续 LLM 主线。
- 不做 Diffusion。
- 不急着进入不可本机复刻的大规模训练。

---

## 27_mlx_gpt_lab_refactor_and_eval_harness

目标：把 19-25 课里分散的 Tiny GPT 代码重构成一个可复用项目。

重点：

- `configs/`
- `src/`
- `scripts/`
- `outputs/runs/`
- `evals/`
- `benchmarks/`
- 统一 train / generate / evaluate / benchmark 入口。

为什么先做它：

现在我们已经理解了模型结构和训练流程，下一步最该补的是工程组织，而不是继续加新概念。

---

## 28_chinese_eval_harness_basics

目标：建立中文小模型评估工具。

重点：

- validation perplexity。
- 固定 prompt 生成。
- 重复率统计。
- 中文标点和乱码检测。
- 简单 QA / 完形填空样例。
- 人工评分表。

原因：

只看 loss 不够。后续所有模型、数据、架构改动都需要可比较的 eval。

---

## 29_mlx_lm_local_inference_intro

目标：从自写 Tiny GPT 过渡到 MLX-LM 真实模型推理。

重点：

- 安装和使用 MLX-LM。
- 加载 MLX community 小模型。
- 使用真实 tokenizer 和 chat template。
- temperature / top-p / top-k。
- 量化模型基本概念。

建议模型：

- Qwen 小尺寸模型的 MLX 版本。
- 或 MLX community 中适合 32GB 统一内存的中文 / 多语种模型。

---

## 30_chat_template_and_instruction_data

目标：理解 base model 和 instruct model 的数据格式差异。

重点：

- chat template。
- system / user / assistant。
- `<bos>` / `<eos>` / special tokens。
- 中文 instruction 数据格式。
- prompt / response packing。

原因：

进入 SFT 之前，必须先理解 tokenizer 和 chat template 如何影响训练目标。

---

## 31_lora_finetuning_with_mlx_lm

目标：用 MLX-LM 做第一次真实模型 LoRA 微调。

重点：

- LoRA 的矩阵分解直觉。
- 冻结 base model。
- 只训练 adapter。
- 准备中文小型 instruction 数据。
- 保存 adapter。
- 加载 adapter 生成。

实践重点：

M1 Pro 上不要做 full fine-tuning，优先 LoRA。

---

## 32_sft_data_quality_lab

目标：理解 SFT 效果主要受数据质量影响。

重点：

- 好坏 instruction 样本对比。
- 去重。
- 过短 / 过长过滤。
- 答案风格统一。
- 中文问答数据规范。
- 小数据过拟合观察。

原因：

SFT 不只是“把 jsonl 塞进去训练”。数据质量比训练步数更关键。

---

## 33_lora_eval_and_regression_testing

目标：给 LoRA 微调建立回归测试。

重点：

- 微调前后固定 prompt 对比。
- 是否遗忘基础能力。
- 是否输出格式稳定。
- 是否出现重复、胡言、模板污染。
- 保存 eval report。

输出：

```text
outputs/evals/before_after_report.md
outputs/evals/prompt_results.jsonl
```

---

## 34_dataset_mixing_and_curriculum_for_small_models

目标：回到 SmolLM3 / OLMo 的启发，做小规模数据 mix 和阶段式训练。

重点：

- 通用中文数据。
- 高质量短文。
- QA / instruction 数据。
- code / math 少量混入。
- staged training vs mixed training。

注意：

仍然保持本地小规模，不追求大模型效果。

---

## 35_reasoning_distillation_intro

目标：理解 DeepSeek-R1 路线里最适合本机实践的一部分：distillation。

重点：

- reasoning trace 数据格式。
- 从强模型生成小规模中文推理样本。
- 用 LoRA 做小模型 reasoning 风格 SFT。
- 评估是否真的提升推理，还是只学会输出长格式。

明确不做：

- 真实 RLHF。
- GRPO 训练大模型。
- 大规模自动数据生成。

---

## 推荐顺序

```text
27 工程重构和 eval harness
28 中文评估基础
29 MLX-LM 本地推理
30 chat template 和指令数据
31 LoRA 微调
32 SFT 数据质量
33 LoRA 评估回归
34 小模型数据 mix / curriculum
35 reasoning distillation 入门
```

## 为什么 27 应该是下一课

我们已经具备：

- tokenizer
- BPE
- Tiny GPT
- KV Cache
- sampling
- scaling
- modern architecture

但工程上还没有统一项目。继续堆新课会增加重复代码。

所以最合理的下一步是：

```text
27_mlx_gpt_lab_refactor_and_eval_harness
```

先把已有知识整理成可复用框架，再进入 MLX-LM 和 LoRA。
