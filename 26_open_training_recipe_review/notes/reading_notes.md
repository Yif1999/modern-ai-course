# 阅读课简记

## 本节应该记住什么

这节不是学习一个新算子，也不是训练一个更大的模型。

这节要建立一个判断框架：

```text
一个真实 LLM 项目，不能只看 model.py。
还要看 data、tokenizer、config、training loop、checkpoint、eval、generate、deployment、post-training。
```

## 七个项目各看什么

- nanoGPT：看最小训练闭环。
- LitGPT：看工程化、多 workflow、LoRA / QLoRA。
- OLMo / OLMo 2：看 fully open recipe 和实验透明度。
- SmolLM3：看小模型训练配方和阶段式训练。
- Qwen / Qwen3：看中文 / 多语种模型生态和 chat template。
- DeepSeek-V3 / R1：看大模型效率架构和 reasoning 后训练路线。
- MLX-LM：看 Apple Silicon 上真实模型推理和 LoRA 实践。

## 对我们最重要的变化

后续不能再让每一课都生成一套新的训练脚本。

应该进入：

```text
统一项目结构
统一配置
统一数据管线
统一模型接口
统一训练入口
统一评估入口
统一输出目录
```

这就是第 27 课应该做的事。
