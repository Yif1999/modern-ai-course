# AI Lab Lessons

这是一个面向学习的 AI / LLM 实验课程目录，主线从张量、自动微分、线性模型、CNN，一路推进到字符级语言模型、Tiny GPT、中文 BPE、MLX 训练工程和本地训练监控。

本仓库定位是：

- 用小实验理解深度学习和 GPT 的核心机制。
- 优先使用中文笔记，降低学习门槛。
- 保留代码可读性，不追求生产级大模型训练平台。
- 数据、模型权重和训练产物不随仓库发布。

## 目录结构

课程按编号组织，例如：

```text
00_environment
01_tensor_autograd
02_linear_regression
...
27_mlx_gpt_lab_refactor_and_eval_harness
28_chinese_fun_corpus_pipeline
29_fun_corpus_continued_pretraining_smoke
30_m1_pro_qwen_like_pretraining_maxout
```

每课通常包含：

- `notes.md`：中文课程笔记。
- `*.py`：实验脚本。
- `configs/`：部分课程的实验配置。
- `reports/` / `diagrams/`：课程报告或图示。
- `assets/`：部分课程内的精选公开教学图。

不会提交到公开仓库的内容包括：

- `data/`：下载数据、清洗数据、tokenized 数据。
- `outputs/`：训练日志、曲线、样本、checkpoint。
- `*.safetensors` / `*.npy` / `*.jsonl` 等实验产物。

## 环境

建议使用 Python 虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

常用依赖包括：

```bash
python -m pip install numpy matplotlib
python -m pip install mlx
python -m pip install torch torchvision
python -m pip install tokenizers datasets transformers
```

部分可视化课程使用 Manim：

```bash
python -m pip install manim
```

第 27 课 Dashboard 包含独立前后端，请分别查看：

```text
27_mlx_gpt_lab_refactor_and_eval_harness/web_backend/README.md
27_mlx_gpt_lab_refactor_and_eval_harness/web_frontend/README.md
```

## 数据说明

本仓库不附带真实数据集、清洗后的语料、tokenized shards 或模型 checkpoint。

原因：

- 数据集通常有各自许可证和使用条款。
- 真实中文网络语料可能包含版权、隐私或安全风险。
- 训练产物体积较大，不适合作为教学源码发布。

请根据每课脚本自行下载或生成数据。详情见 [DATASETS.md](DATASETS.md)。

## 发布状态

如果你要 fork 或重新发布本课程，建议先阅读 [PUBLISHING_CHECKLIST.md](PUBLISHING_CHECKLIST.md)，确认没有误提交数据、权重、缓存、密钥或本机运行产物。
