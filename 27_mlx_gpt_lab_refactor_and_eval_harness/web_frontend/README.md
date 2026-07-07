# MLX GPT Lab Dashboard Frontend

这是 Vite + React 前端，只通过 FastAPI 只读 API 查看训练输出，不会启动、停止或修改训练。

## API 地址

默认 API 会自动使用当前页面的 hostname 和后端端口 `8765`：

```text
http://<当前页面 hostname>:8765
```

例如在本机打开：

```text
http://localhost:5173
```

前端会自动访问：

```text
http://localhost:8765
```

如果在第二台机器打开：

```text
http://192.168.1.10:5173
```

前端会自动访问：

```text
http://192.168.1.10:8765
```

页面右上角也可以手动填写 API 地址，例如：

```text
http://192.168.1.10:8765
```

点击 `Save` 后会保存到浏览器 localStorage。点击 `Auto` 会清除手动配置，恢复自动跟随页面 hostname。

也可以通过 URL 参数临时指定：

```text
http://192.168.1.10:5173/?api=http://192.168.1.10:8765
```

如果需要固定写入构建环境，可以通过 `.env` 配置：

```bash
VITE_API_BASE_URL=http://192.168.1.10:8765
```

## 安装

```bash
cd /Volumes/T7/Dev/ai-lab/27_mlx_gpt_lab_refactor_and_eval_harness/web_frontend
npm install
```

## 本地开发

```bash
npm run dev
```

访问：

```text
http://localhost:5173
```

## 构建

```bash
npm run build
```

## 当前功能

- run 列表
- 当前 run metrics
- loss 曲线
- dataset preview：预览训练样本、数据来源、类别占比
- tokenizer preview：查看样本文本被当前 tokenizer 切成哪些 token
- config 查看
- samples / final text 查看
- eval results 查看
- benchmark 查看
- 自动刷新

## 后续可扩展

- token-level top-k viewer
- logits distribution viewer
- attention heatmap viewer
- prompt playground
- checkpoint comparison
- multi-run comparison
- NAS 部署
