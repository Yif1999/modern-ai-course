# 16：Tiny GPT 生成策略、Temperature、Top-k、Top-p

## 这一课解决什么问题？

上一课我们训练了一个字符级 Tiny GPT。

训练时，模型学习的是：

```text
根据前面的 token，预测下一个 token
```

这一课不改变模型结构，而是观察生成阶段的策略差异。

同一个模型、同一个 prompt，如果使用不同采样策略，生成文本会有不同风格。

---

## 训练和生成的区别

训练时：

```text
输入一整段 token 序列
每个位置都输出 logits
每个位置都和下一个 token 计算 cross entropy
```

所以训练时 logits 的 shape 是：

```text
[batch, seq_len, vocab_size]
```

生成时：

```text
输入当前已有文本
模型输出每个位置的 logits
只取最后一个位置的 logits
采样出下一个 token
把新 token 拼回上下文
重复这个过程
```

因为生成时真正需要的是“下一个 token”，所以只使用：

```text
logits[:, -1, :]
```

---

## logits 和 probabilities 的区别

logits 是模型直接输出的原始分数。

它们不是概率：

1. 可以是负数。
2. 不要求加起来等于 1。
3. 不要求落在 0 到 1 之间。

probabilities 是经过 softmax 后得到的概率分布：

```text
probabilities = softmax(logits)
```

每个 token 都会有一个概率，所有概率加起来等于 1。

---

## softmax 的作用

softmax 会把一组 logits 变成概率分布。

如果某个 token 的 logit 更高，它得到的概率也会更高。

生成时，我们通常不是直接从 logits 采样，而是：

```text
logits
↓
softmax
↓
probabilities
↓
sample next token
```

---

## greedy decoding 是什么？

greedy decoding 的规则很简单：

```text
每一步都选择概率最高的 token
```

优点：

```text
稳定
可复现
不随机
```

缺点：

```text
容易保守
容易重复
容易卡在局部最优的句子模式里
```

它不会探索第二、第三候选 token。

---

## temperature 是什么？

temperature 用来调整概率分布的尖锐程度。

生成时常见写法是：

```text
probabilities = softmax(logits / temperature)
```

temperature 越低，分布越尖锐。

temperature 越高，分布越平坦。

---

## 低 temperature 和高 temperature 的区别

低 temperature，例如：

```text
temperature = 0.5
```

效果是：

```text
高概率 token 更高
低概率 token 更低
生成更保守
更接近 greedy
更容易重复训练文本里的固定模式
```

高 temperature，例如：

```text
temperature = 1.5
```

效果是：

```text
概率分布更分散
低概率 token 也更有机会被选中
生成更随机
更有变化
也更容易出现错字和不自然片段
```

---

## top-k sampling 是什么？

top-k sampling 会只保留概率最高的 k 个 token。

例如：

```text
top_k = 5
```

表示每一步只从概率最高的 5 个 token 里采样。

其他 token 的概率会被设为 0。

这样可以过滤掉大量低概率、明显不合理的 token。

---

## top-p sampling 是什么？

top-p sampling 也叫 nucleus sampling。

它不是固定保留 k 个 token，而是按概率从高到低排序，保留累计概率达到 p 的最小 token 集合。

例如：

```text
top_p = 0.8
```

表示：

```text
从最高概率 token 开始累加
直到累计概率 >= 0.8
只在这些 token 里采样
```

top-p 的候选 token 数量不是固定的。

如果模型很确定，可能只保留 1 到 2 个 token。

如果模型不确定，可能保留更多 token。

---

## top-k 和 top-p 的区别

top-k：

```text
固定保留 k 个 token
```

top-p：

```text
固定保留累计概率 p，对应的 token 数量会变化
```

所以 top-p 更依赖当前概率分布的形状。

在模型很确定时，top-p 会更严格。

在模型不确定时，top-p 会自动放宽候选集合。

---

## 为什么采样策略会影响生成文本质量？

模型输出的是一个概率分布，而不是唯一答案。

采样策略决定了：

```text
是否只选最高概率 token
是否允许低概率 token 出现
是否过滤不靠谱的 token
概率分布要更保守还是更随机
```

所以同一个模型，在不同采样策略下会产生不同文本。

这也是为什么大语言模型推理接口里经常有：

```text
temperature
top_k
top_p
```

这些参数。

---

## 本节 MLX 权重加载方式

上一课保存了：

```text
15_tiny_gpt_training/outputs/tiny_gpt_model.safetensors
```

MLX 的 `nn.Module` 可以直接加载 `.safetensors`：

```python
model.load_weights("tiny_gpt_model.safetensors", strict=True)
```

更底层也可以用：

```python
arrays = mx.load("tiny_gpt_model.safetensors")
```

`mx.load` 会根据扩展名识别 `.npy`、`.npz`、`.safetensors`、`.gguf` 等格式。

---

## 为什么现在还不进入 KV Cache？

KV Cache 是生成加速技巧。

它解决的是：

```text
生成时不要每次重复计算所有历史 token 的 K/V
```

但它不会改变采样策略本身，也不会改变模型训练目标。

本节重点是理解生成策略，所以先不引入 KV Cache。

---

## 下一步可以如何优化 Tiny GPT？

可以继续改进：

1. 增大训练语料。
2. 增大 block_size，让模型看到更长上下文。
3. 增加 num_layers。
4. 增加 n_embd。
5. 增加训练步数。
6. 使用更合理的 tokenizer。
7. 加入 KV Cache 提高生成速度。

---

## 这节课的核心结论

Tiny GPT 训练完成后，模型给出的不是唯一答案，而是下一个 token 的概率分布。

生成策略决定如何从这个分布中选 token。

可以粗略理解为：

```text
greedy：最保守，每次选最大概率
temperature：控制随机性
top-k：固定保留前 k 个候选
top-p：按累计概率动态保留候选
```

采样策略不是训练模型，但它会明显影响生成文本的风格和稳定性。
