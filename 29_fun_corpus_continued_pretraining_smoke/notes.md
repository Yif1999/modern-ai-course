# 29：趣味语料 continued pretraining smoke

## 这一课做了什么？

本课不是高质量训练，而是验证第 28 课的中文趣味语料能否进入第 27 课 MLX GPT Lab 的训练流程。

使用的数据来自：

```text
28_chinese_fun_corpus_pipeline
```

复制到本课目录后使用：

```text
data/raw/fun_corpus.txt
data/processed/train_tokens.npy
data/processed/val_tokens.npy
data/metadata/lab_bpe_tokenizer.json
```

训练使用的是 lab tokenizer，不使用 Qwen tokenizer。

Qwen tokenizer 只用于未来真实 Qwen / MLX-LM / LoRA / SFT 兼容性统计。

---

## 当前训练流程是否顺利？

顺利。

本课跑了两个 run：

```text
baseline_fun_smoke
qwen_dense_tiny_fun_smoke
```

两者都完成了：

1. 读取 train / val tokens。
2. 构建模型。
3. 训练并记录 train loss / val loss。
4. 定期生成样本。
5. 保存 loss_curve.png。
6. 保存 metrics.json。
7. 保存 checkpoint。

checkpoint 文件包括：

```text
best_val_model.safetensors
latest_model.safetensors
final_model.safetensors
```

---

## 训练结果

### baseline_fun_smoke

```text
max_iters: 200
tokens_seen: 204800
final_train_loss: 6.5910
final_val_loss: 7.3270
best_val_loss: 7.3270
```

baseline 主要证明链路通了，不用于判断最终生成质量。

### qwen_dense_tiny_fun_smoke

```text
max_iters: 1000
tokens_seen: 1024000
final_train_loss: 4.9562
final_val_loss: 6.6038
best_val_loss: 6.6038
```

qwen_dense_tiny 的 train loss 和 val loss 都下降，说明趣味语料可以用于 continued pretraining。

---

## `<unk>` token 覆盖率和问题分析

当前 lab tokenizer 统计：

```text
vocab_size: 8192
<unk> token 数: 124604
<unk> token 比例: 7.2205%
```

高频 `<unk>` 片段包括：

```text
。
换行
“
”
《
》
英文字符 t
—
‘
’
部分生僻字、日文假名、特殊符号
```

这说明问题不只是 emoji 或奇怪符号，连中文句号、中文引号、书名号这类常见中文标点也有大量 `<unk>`。

原因是当前 lab tokenizer 复用自前面课程，并不是专门为这批趣味语料训练的 tokenizer。

训练本身不会改变 tokenizer 覆盖率。

如果 tokenizer 把很多标点、网络词、特殊表达变成 `<unk>`，模型就看不到这些细节，自然更难学出完整的中文网络表达风格。

---

## 生成样本文本观察

baseline 的生成文本还很碎，主要是高频词堆叠，例如：

```text
图情包、表情、图片、感觉
```

qwen_dense_tiny 的生成略好一些，已经能看到一些第 28 课语料格式痕迹：

```text
来源
例句
表情包
网络中
幽默效果
```

但文本仍然不自然，原因包括：

1. 训练步数很少。
2. 模型很小。
3. 数据混合了梗解释、弱智吧问答、豆瓣问答，风格不统一。
4. `<unk>` 比例偏高。
5. 当前不是从已有强模型 continued pretraining，而是随机初始化短训练。

---

## 本次趣味语料在模型中的表现

这批数据已经能推动模型学习到部分格式和高频主题。

例如：

```text
来源
例句
表情包
网络
幽默
```

但还没学到稳定、可控的“贴吧老哥 / 中文梗 / 调侃”风格。

当前实验更像是：

```text
流程验证 + 数据可用性检查
```

不是：

```text
风格模型训练完成
```

---

## 如果效果不好，tokenizer 可能是关键因素

本课最重要的诊断结论是：

```text
趣味语料可以训练，但 tokenizer 覆盖率不够理想。
```

如果下一步继续训练仍然生成质量一般，优先考虑：

1. 用通用中文语料 + 趣味语料混合重训 lab BPE tokenizer。
2. 增大 lab tokenizer vocab_size，例如 16k 或 32k。
3. 明确保留中文标点、网络符号、emoji、英数字混排。
4. 重新生成 train_tokens.npy / val_tokens.npy。

---

## 对下一步 SFT / LoRA 的建议

如果进入第 30 课 `30_persona_dialog_sft_toy`，建议区分两条路线：

1. 自训小模型路线：
   - 继续使用 lab tokenizer。
   - 先改善 tokenizer 覆盖。
   - 数据格式可以保持简单 prompt / response。

2. 真实 Qwen / MLX-LM 路线：
   - 使用 Qwen tokenizer。
   - 按 Qwen chat template 构造 SFT 数据。
   - 不把 Qwen tokenizer 直接用于当前 Tiny GPT。

本课已经证明数据流和训练流是通的，可以进入下一课。
