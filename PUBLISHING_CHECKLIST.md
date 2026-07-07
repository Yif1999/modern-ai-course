# 发布前检查清单

在把 `lessons/` 作为公开 Git 仓库发布前，建议按下面顺序检查。

## 1. 确认仓库范围

只发布：

```text
lessons/
```

不要发布：

```text
llm-lab/
.venv/
```

## 2. 检查会被提交的文件

如果还没有初始化仓库：

```bash
cd lessons
git init
git status --short
```

确认 `git status` 中不出现：

- `data/`
- `outputs/`
- `node_modules/`
- `__pycache__/`
- `*.npy`
- `*.jsonl`
- `*.safetensors`
- `*.tar.gz`

## 3. 密钥和隐私扫描

至少运行：

```bash
rg -n --hidden -S "(OPENAI_API_KEY|ANTHROPIC_API_KEY|GITHUB_TOKEN|HF_TOKEN|MODELSCOPE|Bearer |hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|password|secret|api[_-]?key)" .
```

如果命中真实 token、账号、私密 URL、cookie、Authorization header，必须删除或替换成示例值。

本机路径如 `/Volumes/T7/Dev/ai-lab` 通常风险较低，但如果希望更干净，可以在文档里替换为 `<PROJECT_ROOT>`。

## 4. 数据和权重检查

确认不会提交：

```bash
find . -type f \( -name "*.npy" -o -name "*.npz" -o -name "*.safetensors" -o -name "*.jsonl" -o -name "*.pt" -o -name "*.pth" \)
```

这些文件如果出现在 `git status` 里，说明 `.gitignore` 没有覆盖到，或者文件已经被 Git 跟踪过。

## 5. 大文件检查

```bash
find . -type f -size +20M -print
```

公开教学仓库里通常不应该有超过 20MB 的文件。大视频、数据、模型和中间产物应放到 Release、网盘或完全不发布。

## 6. 许可证

发布前需要决定代码许可证，例如：

- MIT：宽松，适合教学代码。
- Apache-2.0：宽松，带专利条款。
- 不添加许可证：默认保留版权，但别人复用不方便。

数据集不随仓库发布；如果 README 中引用数据集，请标注来源和用户需自行遵守原数据许可证。

## 7. 推荐首个提交

确认无误后：

```bash
git add .
git status --short
git commit -m "Initial public lesson release"
```

如果 `git add .` 后发现误加了数据或权重，不要提交，先修 `.gitignore` 或取消暂存。

