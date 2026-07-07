# AI Lab 后续教学计划

## 当前定位

AI Lab 当前已经从基础张量、CNN、Tiny GPT、中文 BPE、真实中文数据、小规模 scaling、现代架构组件，推进到一个教学型 MLX GPT Lab。

后续课程的目标不是一次性做成完整训练平台，而是围绕一条清晰主线继续推进：

```text
中文数据
↓
本地小模型训练理解
↓
风格数据与 SFT
↓
真实 MLX-LM 模型推理 / LoRA
↓
评估、安全、偏好优化
↓
RAG 与本地 agent
```

本项目优先服务于：

1. 在 Apple Silicon / MLX 上理解 LLM 训练和推理机制。
2. 保持课程实验可运行、可观察、可复现。
3. 中文优先，尤其关注中文网络语境、对话和风格数据。
4. 从 toy 实验逐步过渡到 MLX-LM / LoRA / 本地真实模型。

---

## 课程推进原则

1. **先打通流程，再追求效果。**
   每一课先确认数据、训练、评估、输出文件都正常，再讨论质量提升。

2. **小模型用于理解机制，真实模型用于验证应用。**
   自训 Tiny GPT / qwen_dense_tiny 主要用于教学；真实可用效果应逐步转向 MLX-LM 和 LoRA。

3. **数据和 tokenizer 是核心技术债。**
   第 28–29 课已经发现 lab tokenizer 对中文标点、引号、书名号和部分网络表达覆盖不足。后续如果生成效果差，不能只怪模型，需要优先检查 tokenizer 和数据质量。

4. **所有课程输出必须落在课程目录内。**
   不污染项目根目录，不把数据下载到全局 `data/`。

5. **报告和可观察性优先。**
   每次实验都要有 config、log、metrics、loss curve、samples、notes 或 report。

6. **安全边界单独处理。**
   趣味语料、调侃语料和 ToxiCN_MM 可作为学习对象，但安全评估和过滤需要单独课程和专门流程。

7. **课程方针可根据实验结果调整。**
   这里的“微调”指调整课程路线和实验设计，不是模型 fine-tuning。

---

## 已完成的后期主线

### 27_mlx_gpt_lab_refactor_and_eval_harness

工程化骨架搭建。

重点：

- 统一 `configs/`、`src/`、`scripts/`、`outputs/`、`evals/`、`benchmarks/`
- 保留两类模型：
  - `baseline_debug`
  - `qwen_dense_tiny`
- smoke test 验证：
  - 训练
  - 生成
  - 评估
  - benchmark
- 增加本地只读 Dashboard：
  - FastAPI backend
  - Vite + React frontend

核心结论：

第 27 课是后续课程的工程底座。后续实验应尽量复用它的配置、训练循环、输出结构和 Dashboard。

---

### 28_chinese_fun_corpus_pipeline

中文趣味语料数据整理。

重点数据类型：

- 中文梗 / meme
- 弱智吧幽默短句
- 中文闲聊 / 对话
- ToxiCN_MM 作为学习对象

输出：

- JSONL raw data
- 文本 corpus
- tokenized train / val
- 数据统计报告
- lab tokenizer 与 Qwen tokenizer 双轨统计

核心结论：

趣味语料已经可以进入训练流程，但 lab tokenizer 覆盖率不理想。尤其 `<unk>` 涉及中文标点、引号、书名号、换行、部分英文字符和生僻字。

---

### 29_fun_corpus_continued_pretraining_smoke

小规模 continued pretraining 验证。

重点：

- 使用第 28 课趣味语料
- 验证训练流程是否可运行
- 观察 `<unk>` token 覆盖率
- 保存 loss、生成样本、metrics、checkpoint
- smoke test，不追求高质量生成

核心结论：

训练流程已打通，loss 可以下降，生成样本出现了“来源、例句、表情包、网络中”等趣味语料痕迹。但文本仍不自然，原因包括训练步数短、模型小、数据混合风格复杂、tokenizer 覆盖不足。

---

## 路线调整：第 30 课改为本地极限预训练实验

原计划第 30 课进入 persona SFT，但当前实验暴露出一个更底层的问题：

```text
我们的自训基座模型还不够强。
```

因此第 30 课暂时不直接进入 SFT，而是改为：

```text
30_m1_pro_qwen_like_pretraining_maxout
```

目标是在 M1 Pro / 32GB 统一内存上，按更接近企业内部训练实验的方式，尽可能把本地小模型能力推向当前硬件上限。

第 30 课新的重点：

- 使用自训 lab ByteLevel BPE tokenizer 作为自训基座主 tokenizer。
- Qwen tokenizer 退回到未来真实 Qwen / MLX-LM / LoRA 的长度统计和兼容性检查用途。
- 使用 1024 context window。
- 使用 Qwen dense-like decoder-only 架构：
  - RoPE
  - RMSNorm
  - SwiGLU
  - GQA
  - QK-Norm
  - Weight Tying
  - bias 尽量关闭
- 不实现 Qwen3.6 的 Gated DeltaNet / 线性注意力混合结构。
  - 原因：当前 context 只有 1K，线性注意力主要价值在超长上下文，本课优先保持 dense attention 主线清晰。
- 做模型尺寸、吞吐、内存、loss 的 probe。
- 接入 Dashboard 性能监控：
  - tokens/sec
  - step time
  - MLX active / peak / cache memory
  - Metal memory
  - 进程 RSS
  - status.json 实时更新
- 白天跑 probe，晚上再启动更长时间训练。

SFT / persona / LoRA 不取消，只是顺延到基座训练路线更清楚之后。

---

## 后续课程路线

### 30_m1_pro_qwen_like_pretraining_maxout

M1 Pro 上的 Qwen-like 中文基座预训练极限实验。

目标：

- 用自训 lab tokenizer 编码中文通用语料 + 趣味语料。
- 使用 1K context 训练 Qwen dense-like 小模型。
- 用 probe 找出当前机器上可承受的参数规模。
- 记录 tokens/sec、MLX 内存峰值、loss 曲线、生成样本。
- 为夜间长训练选择配置。

验收方向：

- 数据准备输出 lab BPE token ids。
- 至少跑通 120M / 180M 级别 probe。
- Dashboard 能读取第 30 课 run 的 status / metrics / samples。
- 输出生产化训练计划和 probe 报告。

---

### 31_persona_dialog_sft_toy

趣味语料 persona SFT 小实验。

目标：

- 构造“赛博老哥助手” persona
- 实现 response-only loss
- 小规模 SFT toy 训练
- 验证风格保留和安全边界

重点问题：

1. instruction / input / response 如何组织。
2. 为什么 SFT 不应该对 user prompt 计算 loss。
3. persona 风格如何体现在 system prompt 和 response 数据里。
4. toy SFT 和真实 LoRA SFT 的区别。

验收方向：

- 生成 SFT toy 数据
- 跑通 response-only loss
- 保存训练日志和样本
- 对比 pretrain 风格与 persona SFT 风格

---

### 32_style_safety_eval_harness

风格 + 安全评估。

目标：

- 测试生成文本是否保留幽默风格
- 检查攻击性、隐私、重复率
- 输出报告和可选统计图表

重点问题：

1. 什么是风格指标。
2. 如何粗略统计重复率、空输出、乱码、攻击性词。
3. 为什么安全评估要独立于训练。
4. 为什么 toy 评估不能代替真实安全红队。

验收方向：

- `eval_prompts.jsonl`
- `style_eval_results.jsonl`
- `safety_eval_results.jsonl`
- `eval_report.md`
- 可选图表

---

### 33_mlx_lm_local_inference_intro

MLX-LM 本地模型推理。

目标：

- 加载官方 MLX-LM 模型
- 理解本地推理流程
- 测试中文生成
- 初步了解量化、LoRA / 微调接口

重点问题：

1. MLX-LM 和我们自写 Tiny GPT 的关系。
2. tokenizer / chat template / model weights 如何配合。
3. 量化为什么能降低内存占用。
4. 为什么真实模型效果远强于自训 toy model。

验收方向：

- 成功加载一个适合本机的小模型
- 生成中文样本
- 记录推理速度和内存观察
- 写清楚 MLX-LM 与教学代码的边界

---

### 34_lora_chinese_style_finetuning_with_mlx_lm

真实小模型 LoRA 微调。

目标：

- 使用趣味语料 / 老哥 persona 数据做小规模 LoRA
- 验证生成风格变化
- 保存 adapter

重点问题：

1. LoRA 训练的是哪些低秩矩阵。
2. adapter 和 base model 的关系。
3. 为什么 LoRA 比 full fine-tuning 轻。
4. 如何避免训练数据太脏导致模型风格失控。

验收方向：

- LoRA adapter
- 微调前后生成对比
- 简单 eval 报告
- 训练配置记录

---

### 35_chat_template_and_instruction_data

指令数据和 chat template。

目标：

- 理解 system / user / assistant 分段
- 构造中文对话 SFT 数据
- 实现或验证 response-only loss
- 为真实 Qwen / MLX-LM 微调准备数据格式

重点问题：

1. chat template 为什么重要。
2. tokenizer 如何插入 special tokens。
3. 为什么不同模型的 chat template 不通用。
4. SFT 数据如何从 raw dialogue 变成训练样本。

验收方向：

- `sft_dataset.jsonl`
- `chat_template_report.md`
- encode / decode 示例
- response mask 示例

---

### 36_preference_optimization_dpo_toy

偏好优化 DPO toy demo。

目标：

- 构造 chosen / rejected pair
- 理解 policy model / reference model
- 验证 DPO loss 和生成风格变化

重点问题：

1. DPO 和 SFT 的区别。
2. chosen / rejected 数据如何构造。
3. reference model 为什么需要冻结。
4. DPO toy demo 和真实偏好训练的差距。

验收方向：

- toy preference dataset
- DPO loss demo
- 训练日志
- chosen / rejected 生成对比

---

### 37_moe_and_modern_architectures

MoE / GQA / MLA / 长上下文 / 现代架构专题。

目标：

- toy implementation
- 观察内存占用、token routing
- 理解现代架构组件

重点问题：

1. MoE 为什么能增大参数量但不等比例增加每 token 计算量。
2. router / expert / top-k routing 是什么。
3. GQA / MQA 如何减少 KV Cache。
4. MLA 属于更高级的 attention 压缩机制，先理解概念，不急于完整复刻。
5. 长上下文带来的计算和内存压力。

验收方向：

- toy MoE forward
- routing 可视化或日志
- 参数量 / 激活专家数统计
- 架构报告

---

### 38_rag_basics

RAG 文档问答基础。

目标：

- 文档切块
- 向量检索
- top-k retrieval
- 上下文注入

重点问题：

1. RAG 解决的是模型参数知识不足还是上下文补充问题。
2. chunk size / overlap 如何影响检索。
3. embedding model 和 generator model 的分工。
4. 为什么检索质量决定回答上限。

验收方向：

- 本地文档集
- chunk metadata
- retrieval results
- prompt with context
- RAG answer report

---

### 39_local_agent_with_mlx_lm_server

本地 agent + MLX-LM server。

目标：

- 简单工具调用
- 多轮任务
- 本机运行

重点问题：

1. agent 和普通 chat 的区别。
2. 工具 schema 如何定义。
3. 多轮状态如何保存。
4. 本地模型能力不足时，agent 为什么容易出错。

验收方向：

- MLX-LM server 启动
- 简单 tool calling demo
- 多轮任务日志
- agent 局限报告

---

## 阶段性技术债

### Tokenizer 覆盖不足

第 28–29 课发现：

```text
lab tokenizer <unk> 比例约 7.22%
```

高频 `<unk>` 包括中文句号、引号、书名号、换行、部分英文字符和生僻字。

后续可选修复：

1. 用通用中文语料 + 趣味语料混合重训 lab BPE。
2. 增大 vocab_size 到 16k / 32k。
3. 明确保留中文标点、emoji、英数字混排和网络符号。
4. 对比新旧 tokenizer 的 token 数、`<unk>` 比例和训练效果。

### 数据风格混杂

当前趣味语料混合了：

- 梗解释
- meme 图片描述
- 弱智吧问答
- GPT-4 回答
- 豆瓣问答

这会导致模型同时学习解释体、问答体、描述体，风格不够统一。

后续 persona SFT 前，应把数据分成：

```text
pretraining corpus
instruction response data
persona dialogue data
eval prompts
```

### Dashboard 只读监控已可用，但需要统一 run 目录

第 27 课 Dashboard 可通过环境变量读取不同课程的 runs：

```bash
MLX_GPT_LAB_RUNS_DIR=/path/to/outputs/runs
```

后续如果希望长期使用 Dashboard，应考虑统一所有新课程 run 输出，或者提供课程选择机制。

---

## 可调整路线

后续课程可以根据实验结果调整：

1. 如果第 31 课 SFT toy 风格不明显，可以先暂停，补做 tokenizer / 数据质量实验。
2. 如果第 32 课 MLX-LM 推理稳定，可以提前进入 LoRA。
3. 如果真实 LoRA 对设备压力大，可以缩小模型或先做数据格式和 dry run。
4. 如果安全问题突出，可以提前做第 31 课评估，不必等 SFT 后。
5. 如果用户更关注应用，可以把 RAG 和 agent 提前；如果更关注训练机制，可以先做 DPO 和 MoE。

---

## 下一课建议

建议进入：

```text
30_persona_dialog_sft_toy
```

但进入前建议明确：

1. SFT toy 用 lab tokenizer；Qwen tokenizer 只用于未来真实 Qwen LoRA / MLX-LM 兼容性统计。
2. 是否先接受当前 `<unk>` 技术债。
3. persona 数据是从第 28 课自动构造，还是人工写一小批高质量样本。
4. 是否先只做 response-only loss 的机制验证。

推荐做法：

```text
先用小规模人工 + 自动混合样本
先跑通 response-only loss
先验证风格变化
不追求真实可用助手
```
