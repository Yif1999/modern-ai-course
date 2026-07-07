# 第 24 课：M1 Pro 上的中文 Tiny GPT 小规模 Scaling 实验

## 这一课解决什么问题？

第 23 课我们已经用真实中文开源数据训练了一个 Tiny GPT。

补充继续训练实验说明：

```text
train loss 和 val loss 还能继续下降
```

这表示当前模型还没有充分学完当前数据，更像 underfitting / 训练不够，而不是典型过拟合。

第 24 课不引入新模型结构，而是做一组可控 scaling 对比，观察：

```text
数据量
模型大小
上下文长度
训练 token 数
训练速度
```

这些因素如何影响 loss 和生成文本。

---

## 本节目录结构

```text
24_chinese_gpt_scaling_on_m1_pro/
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── small/
│   │   ├── medium/
│   │   └── large/
│   └── cache/
├── outputs/
│   ├── runs/
│   ├── reports/
│   └── plots/
├── prepare_scaling_data.py
├── run_scaling_experiments.py
├── analyze_scaling_results.py
└── notes.md
```

所有数据、cache 和输出都留在本课目录下，不使用项目根目录 `./data`。

---

## 数据策略

本节优先复用第 23 课已经抽样好的真实中文开源语料：

```text
23_chinese_open_dataset_pretraining_run/data/raw/open_zh_corpus.txt
```

以及第 23 课训练好的 BPE tokenizer：

```text
23_chinese_open_dataset_pretraining_run/outputs/tokenizer/chinese_bpe_tokenizer.json
```

本次为了控制网络流量，默认不额外下载更多数据。

实际数据档位：

```text
small:  当前语料的 100 万字符切片
medium: 当前第 23 课完整本地语料
large:  如果本地语料不足 2000 万字符，则只记录未生成原因
```

所以本课的“更多数据”实验是：

```text
small 约 100 万字符
medium 约 466 万字符
```

不是 10M / 20M 的完整大实验。

---

## 四个核心 run

### run_a_baseline

基线实验：

```text
数据：small
block_size: 64
n_embd: 64
num_layers: 2
num_heads: 4
```

它用来作为其它实验的参照。

### run_b_more_data

数据量实验：

```text
数据：medium
其它配置尽量和 baseline 一样
```

观察更多数据是否让 val loss 下降。

### run_c_larger_model

模型容量实验：

```text
n_embd: 128
num_layers: 4
```

观察模型变大后，loss 是否下降，以及训练速度是否明显变慢。

### run_d_longer_context

上下文长度实验：

```text
block_size: 128
其它配置尽量保持 baseline 接近
```

观察更长上下文是否改善 loss 或生成连贯性。

---

## 关键指标怎么看？

### train loss

模型在训练集 batch 上的 next-token cross entropy。

它越低，说明模型越能拟合训练数据。

### val loss

模型在验证集 batch 上的 next-token cross entropy。

它更能反映模型对未参与训练文本的泛化能力。

### overfit gap

```text
overfit_gap = val loss - train loss
```

如果 gap 越来越大，说明模型对训练集学得更多，但对验证集泛化没有同步提升。

### tokens_seen

```text
tokens_seen = steps * batch_size * block_size
```

它表示训练过程中模型大约看过多少 token。

不同 block_size / batch_size / steps 的实验，应该用 tokens_seen 一起比较。

### tokens/sec

每秒处理多少训练 token。

它用于观察不同配置在 M1 Pro 上的速度差异。

---

## 为什么这不是真正的 scaling law？

真正的 scaling law 需要：

```text
更大数据
更多模型尺寸
更充分训练
重复实验
严格控制变量
```

本课只是教学级小规模实验。

它的目标不是得到通用定律，而是建立直觉：

```text
模型变大通常更慢
上下文变长通常更慢
更多数据可能改善 val loss
训练步数不够时 loss 还会继续下降
```

---

## 当前设备上的实用判断

M1 Pro / 32GB 统一内存适合做：

```text
小模型结构实验
tokenizer / 数据管线实验
短时间预训练观察
小规模超参数对比
```

不适合在本课程阶段直接追求高质量中文大模型。

更现实的目标是：

```text
跑通流程
记录指标
理解瓶颈
逐步扩大
```

---

## 本节输出怎么看？

每个 run 的结果在：

```text
outputs/runs/<run_name>/
```

重点看：

```text
config.json
training_log.txt
loss_curve.png
final_generated_text.txt
metrics.json
```

汇总报告在：

```text
outputs/reports/scaling_report.md
outputs/reports/scaling_summary.csv
```

对比图在：

```text
outputs/plots/val_loss_comparison.png
outputs/plots/tokens_per_second_comparison.png
outputs/plots/final_loss_vs_tokens_seen.png
```

---

## 本节核心结论

这节课要看的不是某一个生成样本是否漂亮，而是：

```text
配置变化 → loss 变化
配置变化 → 速度变化
配置变化 → 生成文本变化
```

也就是建立中文 Tiny GPT 在本机上继续扩展时的工程直觉。

---

## 本次实际实验结果

本次实际跑了 4 个 run：

```text
run_a_baseline
run_b_more_data
run_c_larger_model
run_d_longer_context
```

指标如下：

| run | 数据 | block_size | n_embd | layers | tokens_seen | tokens/sec | train loss | val loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| run_a_baseline | small | 64 | 64 | 2 | 1,024,000 | 85,706.7 | 7.0043 | 7.2152 |
| run_b_more_data | medium | 64 | 64 | 2 | 1,024,000 | 80,450.8 | 7.0928 | 7.1984 |
| run_c_larger_model | small | 64 | 128 | 4 | 1,024,000 | 48,842.7 | 7.4759 | 7.5848 |
| run_d_longer_context | small | 128 | 64 | 2 | 2,048,000 | 123,369.3 | 6.3624 | 6.8291 |

注意：

```text
run_d_longer_context 的 block_size 翻倍，所以同样 1000 step 下 tokens_seen 也翻倍。
```

因此它的 val loss 最低，不应该只理解成“上下文更长一定更好”，还包含“它看过更多 token”的影响。

---

## 本次观察

### 数据量增加

`run_b_more_data` 使用 medium 数据，val loss 从 baseline 的 `7.2152` 小幅下降到 `7.1984`。

这说明在这次短训练里，更多数据有一点帮助，但提升不大。

原因可能是：

```text
训练步数太短
模型容量较小
真实数据更杂
```

### 模型变大

`run_c_larger_model` 把模型增大到：

```text
n_embd = 128
num_layers = 4
```

但 val loss 反而变成 `7.5848`，比 baseline 更差。

这不表示大模型一定差，而是说明：

```text
大模型需要更多训练 token
可能需要更合适的 learning rate
短训练下更难充分发挥容量
```

它的速度也明显下降，tokens/sec 约为 baseline 的 `0.57` 倍。

### 上下文变长

`run_d_longer_context` 使用：

```text
block_size = 128
```

最终 val loss 最低，为 `6.8291`。

但它同时看了 `2,048,000` tokens，是其它 run 的两倍。

所以本次更准确的结论是：

```text
更长 block_size + 更多训练 token 在当前实验里最有效
```

不是单独证明长上下文一定更好。

### 生成文本

所有 run 都已经能生成中文片段，并明显带有真实数据主题：

```text
农业
农村
乡村振兴
合作社
政策
产业发展
```

但生成文本仍然有明显问题：

```text
重复较多
标点不稳定
句子逻辑不完整
局部词组像中文，但长程语义弱
```

这说明模型学到了一些局部分布，但还没有形成稳定的长文本能力。

---

## 当前建议

当前 M1 Pro / 32GB 上比较舒服的配置是：

```text
block_size = 128
batch_size = 16
n_embd = 64
num_layers = 2
num_heads = 4
learning_rate = 2e-3
```

如果下一步继续扩展，优先级建议是：

1. 先增加训练步数，让 baseline / longer context 更充分训练。
2. 再扩大数据量，做 10M / 20M 字符数据实验。
3. 最后再扩大模型，例如 n_embd=128、num_layers=4，并配合更长训练和更低 learning rate。

直接把模型变大，但训练 token 不增加，收益不明显，速度还会变慢。

---

## 24b 受控补充实验结果

补充实验只回答两个问题：

```text
1. 大模型上次表现差，是不是因为训练 token 不够？
2. 小模型继续增加数据，是否仍然比放大模型更划算？
```

本次新增两个 run：

| run | 数据 | checkpoint | block_size | n_embd | layers | tokens_seen | tokens/sec | train loss | val loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| followup_a_larger_model_more_training | small | 从 run_c 继续 | 64 | 128 | 4 | 3,072,000 | 48,858.0 | 6.8470 | 7.1738 |
| followup_b_baseline_more_data | medium | 从头训练 | 128 | 64 | 2 | 3,072,000 | 115,540.0 | 6.6191 | 6.6649 |

### 大模型继续训练是否改善？

改善明显。

第 24 课里，大模型 `run_c_larger_model` 的 val loss 是：

```text
7.5848
```

本次从 checkpoint 继续训练后，val loss 变成：

```text
7.1738
```

下降了约：

```text
0.4110
```

这说明上次大模型表现差，很大一部分原因确实是训练 token 不够。

### 大模型是否追上 baseline？

第 24 课 baseline 的 val loss 是：

```text
7.2152
```

本次大模型继续训练后 val loss 是：

```text
7.1738
```

所以它已经略微追上 baseline。

但它的速度只有：

```text
48,858 tokens/sec
```

明显慢于小模型。

### 小模型更多数据是否更划算？

小模型使用 medium 数据、block_size=128，训练到同样 `3,072,000` tokens_seen 后：

```text
val loss = 6.6649
tokens/sec = 115,540.0
```

它比大模型：

```text
loss 更低
速度更快
泛化 gap 更小
```

因此在当前设备和实验规模下，小模型 + 更多数据 + 更多训练 token 仍然比直接放大模型更划算。

### 24b 结论

当前最划算的方向排序是：

1. 增加训练 token。
2. 增加数据量。
3. 适度增加上下文长度。
4. 最后才考虑增大模型。

如果要扩大模型，必须同步增加训练 token，并重新调 learning rate / warmup。

当前已经可以进入下一课：

```text
25_architecture_modernization_lab
```

因为单纯 scaling 已经暴露出结构和优化细节的重要性，下一步可以研究 RMSNorm、SwiGLU、RoPE、weight tying、dropout 等现代 GPT 组件。
