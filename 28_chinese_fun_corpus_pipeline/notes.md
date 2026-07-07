# 28：中文趣味语料数据管线

## 这一课解决什么问题？

前面我们已经能用中文开源语料训练 Tiny GPT / qwen_dense_tiny。

这一课不训练模型，而是单独整理一批“趣味中文语料”，为后续 continued pretraining 做准备。

目标语料风格包括：

- 中文梗解释
- 弱智吧式幽默问答
- 网络吐槽 / 调侃文本
- 少量中文闲聊 / 豆瓣问答

最终输出不是模型，而是可训练的 token 数据：

```text
fun_corpus.txt
train_tokens.npy
val_tokens.npy
train_val_metadata.json
```

---

## 实际接入的数据源

本课启用的数据源包括：

| 数据源 | 类别 | 实际保留 |
|---|---|---:|
| CHIME | meme_explanation | 1458 |
| REILX/chinese-meme-description-dataset | meme_explanation | 2000 |
| LooksJuicy/ruozhiba | humor_ruozhiba | 1495 |
| hfl/ruozhiba_gpt4 | humor_ruozhiba | 2000 |
| m-a-p/COIG-CQIA:douban | chat_dialogue | 1169 |
| JunyuLu/ToxiCN_MM | social_fun | 0 |

`JunyuLu/ToxiCN_MM` 已按要求启用和探测，但当前 Hugging Face 可见字段只有 `image`，没有可直接用于文本预训练的 caption、文本或标签字段。

所以它被记录在数据源统计中，但没有进入 `fun_corpus.txt`。

---

## 清洗与统计策略

本课清洗做得比较保守：

1. 去掉空文本。
2. 去掉过短文本。
3. 去掉过长文本。
4. 去掉重复文本。
5. 估算中文比例，过滤明显不是中文主体的文本。
6. 不做安全过滤。

一个重要修正：

本课现在使用：

```python
unicodedata.normalize("NFC", text)
```

而不是：

```python
unicodedata.normalize("NFKC", text)
```

原因是 `NFKC` 会把部分全角中文标点折叠成半角英文标点，例如：

```text
， -> ,
！ -> !
？ -> ?
```

这会破坏中文网络文本原有的标点风格。

所以当前清洗不会把中文标点改成英文标点。

同时，为了避免模型学到错误排版，清洗会额外处理三类中文空格噪声：

```text
你好， 世界 -> 你好，世界
我 在 重庆 吃 火锅 -> 我在重庆吃火锅
昨天 8 度 -> 昨天8度
用 MLX 训练 -> 用MLX训练
MacBook Pro -> MacBook Pro
```

也就是：保留中文标点本身，但去掉中文标点前后的异常半角空格，去掉中文字符之间由分词或转写引入的空格，也去掉中文和数字 / 英文 token 边界上的空格。英文短语内部的正常空格会保留，例如 `MacBook Pro`。

安全边界说明：

本课只做数据管线，不做安全评估，也不对调侃、冒犯、毒性内容做价值判断。后续如果要用于面向用户的模型，需要专门增加安全数据治理和评估流程。

---

## 本次样本统计

本次生成的语料统计为：

```text
清洗后样本数：8122
总字符数：2069752
平均长度：254.83
中文比例估计：0.9774
字符级 token 数：2085994
BPE token 数：1725701
train tokens：1553130
val tokens：172571
```

类别占比：

```text
humor_ruozhiba：43.03%
meme_explanation：42.58%
chat_dialogue：14.39%
```

这说明当前 corpus 的主体是：

```text
中文梗解释 + 弱智吧幽默问答
```

闲聊对话只占较小比例。

---

## Tokenization 方式

本课现在有两套 tokenizer 统计：

```text
lab tokenizer
Qwen tokenizer
```

---

## lab tokenizer

lab tokenizer 用于我们自己的 MLX GPT Lab / qwen_dense_tiny 小模型训练。

本课复用第 23 课训练好的中文 BPE tokenizer：

```text
23_chinese_open_dataset_pretraining_run/outputs/tokenizer/chinese_bpe_tokenizer.json
```

处理流程是：

```text
fun_corpus.txt
↓
BPE tokenizer.encode(...)
↓
token ids
↓
train / val split
↓
train_tokens.npy / val_tokens.npy
```

训练时模型看到的是 token id，不是原始字符串。

对应输出：

```text
data/processed/train_tokens.npy
data/processed/val_tokens.npy
data/metadata/lab_tokenizer_metadata.json
```

---

## Qwen tokenizer

Qwen tokenizer 只用于未来真实 Qwen / MLX-LM / LoRA / SFT 的兼容性统计。

本课会优先尝试：

```text
Qwen/Qwen3.6-27B
```

如果失败，再 fallback 到：

```text
Qwen/Qwen3-0.6B
Qwen/Qwen2.5-0.5B
```

本次实际成功加载：

```text
Qwen/Qwen3.6-27B
```

注意：

这里只加载 tokenizer，不加载模型权重。

Qwen tokenizer 的作用是回答：

1. 这批趣味语料在真实 Qwen tokenizer 下会变成多少 token。
2. 是否有超长样本。
3. 是否能识别 Qwen chat special tokens。
4. 未来是否适合转成 Qwen LoRA / SFT 数据。

它不用于当前 Tiny GPT 训练。

原因是 Qwen tokenizer 的 vocab 很大。本次加载到的 `Qwen/Qwen3.6-27B` vocab_size 为：

```text
248077
```

如果直接拿它训练当前小模型，LM Head 会变成：

```text
n_embd -> 248077
```

这会显著增加参数量和训练难度，不适合我们的教学型小模型。

对应输出：

```text
data/metadata/qwen_token_stats.json
outputs/reports/qwen_tokenizer_compat_report.md
outputs/reports/tokenizer_dual_track_report.md
```

---

## 一个重要观察：`<unk>` 比例偏高

本次 BPE 编码后：

```text
<unk> token 数：124604
<unk> token 比例：7.2205%
```

这说明复用第 23 课 tokenizer 时，有一部分趣味语料中的字符、符号或网络表达没有被很好覆盖。

这不影响本课数据管线验收，但对下一课训练有影响：

- `<unk>` 太多会损失原文信息。
- 网络梗、emoji、特殊符号、外文混排更容易变成未知 token。
- 如果 continued pretraining 效果不好，tokenizer 覆盖率会是一个重点诊断对象。

下一步可以考虑：

1. 继续复用当前 tokenizer，先观察训练效果。
2. 用中文通用语料 + 趣味语料混合重训 BPE tokenizer。
3. 增大 vocab_size。
4. 明确保留常见网络符号、emoji、标点和英数字混排。

---

## train / val 划分规则

本课把完整 token 序列按比例切分：

```text
90% train
10% val
```

输出：

```text
train_tokens.npy
val_tokens.npy
```

后续训练时：

```text
x = tokens[start : start + block_size]
y = tokens[start + 1 : start + block_size + 1]
```

也就是继续做 next-token prediction。

---

## 样本预览

样本文件保存在：

```text
outputs/samples/
```

包括：

```text
meme_explanation_samples.txt
humor_ruozhiba_samples.txt
chat_dialogue_samples.txt
mixed_samples.txt
```

这些文件用于人工检查语料风格是否符合预期。

---

## 未来 continued pretraining 使用计划

下一课可以直接读取：

```text
data/processed/train_tokens.npy
data/processed/val_tokens.npy
```

用第 27 课的工程化训练框架，或者单独脚本，对已有 qwen_dense_tiny / Tiny GPT 继续训练。

重点观察：

1. loss 是否下降。
2. 生成文本是否更像中文网络梗和幽默短句。
3. 是否出现重复、乱码、过拟合。
4. `<unk>` 偏高是否明显影响生成质量。

---

## 本课核心结论

这一课不是为了让模型立刻变强，而是为了把“中文趣味语料”整理成可训练数据。

完成后，我们已经有了：

```text
原始 JSONL
清洗后 corpus
BPE token ids
train / val split
数据报告
样本预览
```

这就是后续 continued pretraining 的输入基础。
