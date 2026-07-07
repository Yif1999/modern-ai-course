# 开源 LLM 训练项目与训练配方总览

本节不是为了直接复刻这些项目，而是建立一张工程地图：真实 LLM 项目通常怎样组织数据、tokenizer、模型架构、训练循环、checkpoint、评估、推理和后训练。

参考链接统一记录在 `26_open_training_recipe_review/references/links.md`。

---

## 1. nanoGPT

### 定位

nanoGPT 是极简 GPT 训练项目。它把 GPT 训练最核心的东西压缩在少数几个文件里：

- `model.py`：GPT 模型定义。
- `train.py`：训练循环。
- `sample.py`：采样生成。
- `data/`：Shakespeare、OpenWebText 等数据准备示例。
- `config/`：用 Python 配置文件切换实验设置。

### 解决什么问题

它解决的是“我想从最少代码看懂 GPT 训练到底怎么跑”的问题。

它适合学习：

- tokenizer 后的 token ids 如何进入模型。
- `block_size`、`batch_size`、`n_layer`、`n_head`、`n_embd` 如何组织。
- forward、loss、backward、optimizer、checkpoint、sample 的最小闭环。
- 一个训练脚本如何同时支持从零训练、恢复训练、加载 GPT-2 权重、采样。

### 项目结构特点

nanoGPT 的结构很扁平，教学价值很高。它不是复杂框架，而是直接把关键逻辑放在少数文件中。

对我们最有用的是：

- `train.py` 的主训练流程。
- `config/*.py` 的配置覆盖方式。
- `data/*/prepare.py` 的数据预处理思路。
- `sample.py` 的生成入口。

### 数据和 tokenizer

nanoGPT 的教学路径通常从字符级 Shakespeare 开始，再到 OpenWebText + `tiktoken` BPE。数据最终会被处理成连续 token id 文件，例如 `train.bin` / `val.bin`。

这个思路和我们本地课程已经做过的：

```text
raw text -> tokenizer -> train_tokens.npy / val_tokens.npy -> batch
```

是同一类工程抽象。

### 不适合本机复刻的部分

完整复现 GPT-2 级别训练需要多卡高端 GPU。M1 Pro 适合复刻它的结构和小数据训练流程，不适合追求它的中大型训练结果。

### 对我们课程的启发

我们的 MLX GPT Lab 应该保留 nanoGPT 的清晰性：

- 一个清楚的 `train.py`
- 一个清楚的 `model.py`
- 简单配置入口
- 数据预处理和训练解耦
- 采样脚本独立

---

## 2. LitGPT

### 定位

LitGPT 是更工程化的 LLM 项目，目标不是只教学，而是覆盖从预训练、微调、推理到部署的完整工作流。

它强调：

- 从零实现主流 LLM 架构。
- 支持 pretrain / finetune / deploy。
- 支持 LoRA、QLoRA、adapter。
- 支持量化和低显存训练。
- 支持从单机到多机训练。
- 用配置和 recipe 管理实验。

### 解决什么问题

LitGPT 解决的是“真实工程中怎样组织多模型、多任务、多训练方式”的问题。

它适合学习：

- 项目如何从脚本走向 CLI / recipe。
- 训练、微调、推理、评估如何拆分。
- LoRA / QLoRA / adapter 的工程入口怎么设计。
- 多模型架构如何共用一套训练和推理基础设施。

### 项目结构特点

LitGPT 比 nanoGPT 更大，但仍强调“无过度抽象”和“从零实现”。它的价值不在于某一个训练循环，而在于工程边界：

```text
configs / recipes
model implementations
pretrain
finetune
generate / chat
quantization
evaluation
deployment
tests
```

### 数据和 tokenizer

LitGPT 需要适配不同模型家族的 tokenizer 和 chat template。它更接近真实开源模型使用方式：不是自己随手训练一个字符 tokenizer，而是围绕模型家族已有 tokenizer、checkpoint 和数据格式工作。

### 不适合本机复刻的部分

LitGPT 的大模型预训练、多卡训练和完整部署链路不适合 M1 Pro 本机完整复刻。但它非常适合阅读和借鉴工程结构。

### 对我们课程的启发

后续进入 SFT / LoRA 时，我们应该借鉴 LitGPT：

- `scripts/` 管理命令入口。
- `configs/` 管理实验配置。
- `data/` 和 `tokenizer/` 解耦。
- `outputs/runs/<run_name>/` 保存每次实验。
- `evals/` 和 `benchmarks/` 作为独立模块。

---

## 3. OLMo / OLMo 2

### 定位

OLMo 的核心价值是 fully open training recipe。它不只是放出模型权重，而是尽量开放训练数据、训练代码、训练日志、评估结果、中间 checkpoint 和数据工具。

OLMo 2 继续强调：

- open and accessible training data
- open-source training code
- reproducible training recipes
- transparent evaluations
- intermediate checkpoints

### 解决什么问题

OLMo 解决的是“怎样让 LLM 训练成为可研究、可复现、可审计的科学工程”的问题。

它适合学习：

- 训练报告应该记录什么。
- 数据 mix / staged training 如何描述。
- 中间 checkpoint 为什么重要。
- 训练日志、评估日志、数据 manifest 为什么不能省。
- 如何把模型训练从“黑盒调参”变成“可追溯实验”。

### 数据管线

OLMo 生态里最重要的数据概念是 Dolma / Dolmino 这类数据配方。Dolma 是大规模开放语料，Dolmino 更偏高质量、目标能力强化的数据 mix。

OLMo 2 的训练是阶段式的：

```text
Stage 1: 大量 web-based 数据
Stage 2: 更小但质量更高、目标更明确的数据
```

这对我们很重要：真实预训练不是“随便堆文本”，而是有数据来源、清洗、去重、质量过滤、mix 比例和阶段安排。

### 不适合本机复刻的部分

OLMo 的真实训练规模是数万亿 token 和大规模集群。M1 Pro 只能复刻：

- 数据报告格式。
- 小规模数据 manifest。
- run config。
- checkpoint 命名。
- eval harness。
- staged training 的小型模拟。

### 对我们课程的启发

我们最应该借鉴 OLMo 的不是模型规模，而是实验透明度：

- 每次 run 都要有 `config.json`。
- 每个数据集都要有 `metadata.json` 和 `manifest.json`。
- 保存训练日志和 loss 曲线。
- 保存中间 checkpoint。
- 报告中说明数据来源、过滤规则、token 数、训练步数、tokens/sec。

---

## 4. SmolLM3

### 定位

SmolLM3 是 Hugging Face 的小模型训练 recipe。它关注 3B 级别小模型如何通过架构、数据、长上下文和 post-training 做出强表现。

官方资料强调：

- 3B 模型。
- 公开训练 blueprint。
- 多阶段 pretraining。
- 多语种。
- long context。
- reasoning / dual mode。
- GQA、NoPE、intra-document masking 等结构或训练技巧。

### 解决什么问题

SmolLM3 解决的是“参数量不巨大时，如何用更好的 recipe 做强小模型”的问题。

它适合学习：

- 小模型不是只靠堆参数，数据配方和训练阶段很重要。
- 长上下文训练需要专门的数据和 mask 策略。
- GQA 可以减少 KV Cache 成本。
- post-training 可以显著改变模型能力边界。
- 小模型训练也需要完整工程记录。

### 对我们课程的启发

我们后续如果继续走本地中文 Tiny GPT，SmolLM3 的启发是：

- 保留 small model 路线，不急着追大参数。
- 用更好的数据 mix 和阶段式训练提升效果。
- 先做 base model，再做 instruction / reasoning 方向的 post-training。
- 引入 GQA / long-context 实验时，要同时观察 KV Cache 和训练样本组织方式。

### 不适合本机复刻的部分

3B 参数、万亿 token 级训练不适合 M1 Pro。适合复刻的是缩小版 recipe：

```text
小模型
中文数据 mix
阶段式训练
GQA 概念实验
长上下文小样本实验
生成和 eval 对比
```

---

## 5. Qwen / Qwen3

### 定位

Qwen 是中文 / 多语种 LLM 生态里非常重要的开源模型系列。Qwen3 包含 dense 和 MoE 模型，并明确区分 thinking / non-thinking、base / instruct / coder 等路线。

Qwen3 的价值在于：

- 中文和多语种生态成熟。
- 模型尺寸跨度大。
- dense 与 MoE 并存。
- 支持 thinking / non-thinking 模式。
- 部署生态丰富，包括 Transformers、vLLM、SGLang、Ollama、LMStudio、MLX、llama.cpp 等。

### 解决什么问题

Qwen 解决的是“中文 / 多语种模型如何形成完整模型生态”的问题。

它适合学习：

- 中文 tokenizer 和 chat template。
- base / instruct / coder / reasoning 模型分工。
- dense 与 MoE 模型的工程差异。
- 大模型如何围绕推理、代码、聊天、数学能力形成分支。

### 对我们课程的启发

Qwen 对我们最有用的是后续真实模型阶段：

- 用 Qwen 小尺寸模型做 MLX 本地推理。
- 学习中文 tokenizer 和 chat template。
- 学习 instruct 数据格式。
- 后续用 MLX-LM 做 LoRA 微调时，可以优先考虑 Qwen 系小模型。

### 不适合本机复刻的部分

Qwen3 的主训练不适合本机复刻。MoE、长上下文和大规模 post-training 适合阅读，不适合作为本机从零实现目标。

---

## 6. DeepSeek-V3 / DeepSeek-R1

### 定位

DeepSeek-V3 是大规模高效率训练与推理架构的代表，核心关键词包括：

- MoE
- MLA
- DeepSeekMoE
- auxiliary-loss-free load balancing
- multi-token prediction
- 大规模预训练、SFT、RL

DeepSeek-R1 则是 reasoning 后训练路线的代表，核心关键词包括：

- RL for reasoning
- R1-Zero
- cold-start data
- SFT + RL 多阶段 pipeline
- distillation
- 长 CoT

### 解决什么问题

DeepSeek-V3 解决的是“怎样用更高效的架构训练和推理超大模型”的问题。

DeepSeek-R1 解决的是“怎样通过后训练激发 reasoning 能力”的问题。

### 对我们课程的启发

当前阶段不应该实现 DeepSeek-V3 / R1，但应该建立概念地图：

- MLA 是 KV Cache / attention 效率方向的重要概念。
- MoE 是参数规模和计算量解耦的重要方向。
- MTP 提醒我们训练目标也可以升级。
- R1 说明 reasoning 能力主要不是靠 base pretraining 一步完成，而是依赖后训练 pipeline。
- distillation 是把大模型能力迁移到小模型的重要路径。

### 不适合本机复刻的部分

DeepSeek-V3 / R1 的核心训练规模远超 M1 Pro。适合阅读：

- 架构思想。
- 后训练阶段划分。
- RL reasoning 概念。
- distill 数据思路。

不适合本机实现：

- 从零训练 MoE 大模型。
- 复刻 R1 级 RL pipeline。
- 大规模 reasoning 数据生成。

---

## 7. MLX-LM

### 定位

MLX-LM 是 Apple Silicon 上最贴近我们后续实践的真实模型工具链。它支持：

- 本地文本生成。
- 加载 Hugging Face / MLX community 模型。
- 量化。
- LoRA 和 full fine-tuning。
- Python API 和 CLI。
- Apple Silicon 上的 MLX 后端。

### 解决什么问题

它解决的是“如何在 Mac 上真正运行和微调现成 LLM”的问题。

它适合学习：

- 本机真实模型推理。
- 量化模型加载。
- LoRA 微调数据格式。
- 本地 chat / generate。
- 后续从 toy GPT 过渡到真实模型。

### 对我们课程的启发

我们自己从零写 Tiny GPT 是为了理解底层。之后要做真实模型实践，应该切换到 MLX-LM：

- 使用真实 tokenizer。
- 加载真实 checkpoint。
- 做 LoRA 微调。
- 做本地评估和推理服务。

### 不适合本机复刻的部分

MLX-LM 不适合拿来替代基础课的从零实现，因为它抽象掉了很多底层结构。但在“真实模型阶段”，它会是最重要的实践工具。

---

## 总体结论

这些项目可以分成四类：

| 类型 | 代表项目 | 我们应该学什么 |
|---|---|---|
| 极简教学 | nanoGPT | 最小训练闭环、脚本组织、数据到 loss 的路径 |
| 工程框架 | LitGPT | 多工作流、多配置、微调、部署、评估的项目结构 |
| fully open recipe | OLMo / OLMo 2 | 数据、日志、checkpoint、评估、复现性的完整记录 |
| 小模型 recipe | SmolLM3 | 小模型训练路线、阶段式训练、长上下文、GQA |
| 中文 / 多语种生态 | Qwen / Qwen3 | 中文 tokenizer、chat template、instruct、reasoning、部署生态 |
| 大规模效率和 reasoning | DeepSeek-V3 / R1 | MLA、MoE、RL reasoning、distillation 的方向 |
| Apple Silicon 实践 | MLX-LM | 本机推理、量化、LoRA、真实模型实践 |

对我们的课程来说，下一步不是盲目扩大模型，而是把已有 MLX GPT Lab 重构成更像真实项目的结构：

```text
configs/
data/
src/
scripts/
outputs/runs/
evals/
benchmarks/
tokenizer/
checkpoints/
reports/
```

然后再进入更真实的中文评估、LoRA 和本地推理阶段。
