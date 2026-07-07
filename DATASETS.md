# 数据集说明

本课程仓库默认不发布数据集本体。

## 为什么不提交数据？

课程中使用过几类数据：

- MNIST / CIFAR-10 等公开教学数据。
- 中文 toy text。
- Hugging Face / ModelScope 上的中文开源文本数据。
- 中文趣味语料、对话语料、评论语料和 ACG 相关文本。
- tokenizer 训练结果、tokenized `.npy` 分片、checkpoint。

即使某些数据可以公开下载，也不建议直接提交到课程仓库：

1. 数据文件体积大，会让仓库难以 clone。
2. 不同数据集许可证不同，需要用户自行确认。
3. 中文网络语料可能包含用户生成内容、版权文本、隐私片段或不适合直接再分发的内容。
4. 训练样本和模型生成样本可能复现原始数据片段。

## 推荐做法

公开仓库只保留：

- 数据下载 / 抽样 / 清洗脚本。
- 数据源配置示例。
- 数据统计和处理流程说明。
- 小型 toy text 自动生成逻辑。

不保留：

- `data/raw/`
- `data/processed/`
- `data/cache/`
- `outputs/runs/`
- `outputs/checkpoints/`
- `*.npy`
- `*.jsonl`
- `*.safetensors`

## 使用者如何复现？

进入对应课程目录，运行该课的数据准备脚本。例如：

```bash
cd 20_chinese_open_text_pretraining_dataset
python prepare_chinese_dataset.py
```

或：

```bash
cd 23_chinese_open_dataset_pretraining_run
python sample_open_chinese_dataset.py --offline-fallback
```

涉及真实开源数据集的课程，请先阅读脚本中的数据源说明，并确认本地网络、磁盘和数据许可条件。

## 第 28-30 课特别说明

第 28-30 课涉及中文趣味语料、对话、评论、ACG、论坛风格和大规模训练数据准备。

这些课程适合公开：

- 数据管线代码。
- 清洗规则。
- tokenizer / training 代码。
- 配置模板。
- 中文笔记。

不适合公开：

- 实际拉取下来的语料。
- 清洗后的 corpus。
- tokenized shards。
- 训练样本预览。
- 模型 checkpoint。
- 训练输出样本。

