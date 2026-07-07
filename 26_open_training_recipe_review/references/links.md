# 参考链接

本文件只记录官方 GitHub、官方文档、官方博客、模型卡或技术报告入口。正文报告不复制大段原文，只按课程目标做工程视角总结。

## nanoGPT

- GitHub: https://github.com/karpathy/nanoGPT  
  极简 GPT 训练/采样项目，核心文件是 `train.py`、`model.py`、`sample.py` 和 `data/` 里的准备脚本。

## LitGPT

- GitHub: https://github.com/Lightning-AI/litgpt  
  Lightning AI 的 LLM 训练、微调、推理、部署项目，支持 pretrain、finetune、LoRA、QLoRA、adapter、量化等工程流程。

## OLMo / OLMo 2

- OLMo GitHub: https://github.com/allenai/OLMo  
  OLMo 训练、评估、推理代码入口。README 指向更新的 OLMo-core。
- OLMo-core GitHub: https://github.com/allenai/OLMo-core  
  OLMo 生态的新训练基础库。
- OLMo 2 官方页面: https://allenai.org/olmo2  
  OLMo 2 fully-open 模型、训练数据、训练代码、recipe、评估和中间 checkpoint 说明。
- OLMo 2 博客: https://allenai.org/blog/olmo2  
  OLMo 2 的阶段式训练、评估和 fully open 定位说明。
- Dolma GitHub: https://github.com/allenai/dolma  
  OLMo 训练数据 Dolma 的数据工具与说明。
- Dolma 数据集: https://huggingface.co/datasets/allenai/dolma  
  公开预训练语料入口。

## SmolLM3

- Hugging Face 官方博客: https://huggingface.co/blog/smollm3  
  SmolLM3 的模型定位、训练 recipe、三阶段预训练、GQA、NoPE、长上下文和 reasoning 说明。
- Model card: https://huggingface.co/HuggingFaceTB/SmolLM3-3B  
  SmolLM3-3B 模型卡。
- GitHub: https://github.com/huggingface/smollm  
  Hugging Face Smol 模型族入口。

## Qwen / Qwen3

- Qwen3 GitHub: https://github.com/QwenLM/Qwen3  
  Qwen3 系列模型、推理示例、thinking / instruct 版本说明。
- Qwen3 官方博客: https://qwenlm.github.io/blog/qwen3/  
  Qwen3 dense / MoE 模型族、上下文长度、thinking/non-thinking 模式和部署生态说明。

## DeepSeek-V3 / DeepSeek-R1

- DeepSeek-V3 GitHub: https://github.com/deepseek-ai/DeepSeek-V3  
  DeepSeek-V3 模型、MLA、DeepSeekMoE、MTP、训练效率和模型下载入口。
- DeepSeek-V3 技术报告: https://arxiv.org/abs/2412.19437  
  DeepSeek-V3 架构与训练技术报告。
- DeepSeek-R1 GitHub: https://github.com/deepseek-ai/DeepSeek-R1  
  DeepSeek-R1-Zero、DeepSeek-R1、RL reasoning、distill 模型说明。
- DeepSeek-R1 model card: https://huggingface.co/deepseek-ai/DeepSeek-R1  
  DeepSeek-R1 模型卡和模型入口。

## MLX-LM

- GitHub: https://github.com/ml-explore/mlx-lm  
  Apple Silicon 上用 MLX 运行、量化、生成和微调 LLM 的官方项目。
- MLX Community: https://huggingface.co/mlx-community  
  MLX 兼容模型权重社区入口。
