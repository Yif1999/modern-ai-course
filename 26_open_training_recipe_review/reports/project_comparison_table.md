# 开源项目对比表

| 项目名 | 主要框架 | 训练 / 推理 / 微调支持 | 是否适合初学阅读 | 是否适合 M1 Pro 本机实践 | 数据管线特点 | tokenizer 特点 | 架构特点 | 评估支持 | 我们应该借鉴什么 |
|---|---|---|---|---|---|---|---|---|---|
| nanoGPT | PyTorch | 训练、finetune、sample | 非常适合 | 适合小数据、小模型实践 | `data/*/prepare.py` 把 raw text 转成 token bin | 字符级示例 + `tiktoken` BPE | GPT-2 风格 decoder-only Transformer | 简单 train/val loss，非完整 eval harness | 极简 `train.py`、`model.py`、`sample.py`、config 思路 |
| LitGPT | PyTorch / Lightning Fabric | pretrain、finetune、LoRA、QLoRA、adapter、deploy、inference | 适合进阶阅读 | 适合运行小模型推理/部分微调，不适合大训练 | recipe / config 管理不同任务和数据 | 适配多模型 tokenizer 和 chat template | 支持多个主流 LLM 架构 | 有更工程化的评估与部署支持 | CLI、recipe、configs、微调流程、量化和部署边界 |
| OLMo / OLMo 2 | PyTorch / OLMo-core | pretrain、fine-tune、inference、eval | 适合研究型阅读 | 只适合 tiny 示例或阅读，不适合复刻真实规模 | Dolma / Dolmino / OLMo-mix，强调数据来源、阶段式训练、manifest | 真实大模型 tokenizer，强调可复现 | OLMo 系列 decoder-only 架构，关注训练 recipe | 透明评估、中间 checkpoint、训练日志 | fully open 的报告格式、数据记录、checkpoint 和 eval 透明度 |
| SmolLM3 | Hugging Face 生态 | pretraining recipe、instruct / reasoning model | 适合小模型路线阅读 | 不适合从零复刻 3B，但适合借鉴缩小 recipe | 三阶段预训练，公开数据 mixture 和工程 blueprint | 多语种 tokenizer，适合小模型长上下文 | 3B、GQA、NoPE、长上下文、intra-document masking | 官方 benchmark 和模型卡 | 小模型训练 recipe、阶段式训练、GQA、长上下文数据策略 |
| Qwen / Qwen3 | Transformers / 多部署后端 | inference、chat、instruct、coder、reasoning，训练 recipe 非完全开放 | 适合中文生态阅读 | 适合用 MLX / llama.cpp / Ollama 推理小模型，不适合从零训练 | 训练数据不完全开放，生态侧更强 | 中文 / 多语种 tokenizer、chat template、thinking tag | dense + MoE、GQA、长上下文、thinking / non-thinking | 官方 benchmark、模型卡和部署文档 | 中文 tokenizer、chat template、模型分支、后训练产品形态 |
| DeepSeek-V3 | PyTorch / 自研训练系统 | pretrain、SFT、RL，推理代码和权重入口 | 适合架构和训练效率阅读 | 不适合本机复刻 | 14.8T token 级别训练，数据细节只适合概念学习 | 大模型 tokenizer，服务于 V3/R1 生态 | MoE、MLA、DeepSeekMoE、MTP、aux-loss-free load balancing | 技术报告和 benchmark | MLA、MoE、MTP、训练效率设计 |
| DeepSeek-R1 | DeepSeek 生态 | reasoning 后训练、RL、distill | 适合后训练路线阅读 | 适合运行蒸馏小模型，不适合复刻 RL pipeline | reasoning 数据、cold-start、RL、SFT 多阶段 | R1 / distill 模型 tokenizer 设置需要按官方说明 | 基于 V3-Base，重点在 RL reasoning pipeline | 数学、代码、reasoning benchmark | 后训练阶段划分、distillation、reasoning 数据思路 |
| MLX-LM | MLX | generate、chat、quantize、LoRA、full fine-tuning、server / API | 适合实践阅读 | 非常适合 | 接 Hugging Face / 本地数据，主要服务微调和推理 | 复用真实模型 tokenizer 和 chat template | 使用已适配 MLX 的真实模型架构 | 可接本地评估脚本 | Apple Silicon 真实模型推理、量化、LoRA、从 toy 到真实模型的桥 |

## 阅读优先级建议

1. 想看懂最小训练闭环：先读 nanoGPT。
2. 想看懂工程化：读 LitGPT。
3. 想看懂 fully open recipe：读 OLMo / OLMo 2。
4. 想看小模型怎么做强：读 SmolLM3。
5. 想做中文真实模型实践：看 Qwen + MLX-LM。
6. 想理解前沿大模型效率和 reasoning：读 DeepSeek-V3 / R1，但暂不复刻。
