# pi05 Runtime Notes

训练命令是在 `robotwin` shell 里执行的，但实际运行 `pi05/scripts/train.py` 的解释器是 `pi05/.venv/bin/python`。

之前服务器上的实际路径是：

```bash
<workspace>/conda_envs/robotwin/bin/python
<repo>/pi05/.venv/bin/python
```

其中：

```text
robotwin conda: Python 3.10
pi05 .venv:     Python 3.11
```

因此复现时建议：

1. 先进入 `robotwin` 环境，保持系统库、CUDA、ffmpeg 等环境一致。
2. 在 `pi05` 目录下用 `uv sync` 或现有 `.venv` 方案创建 Python 3.11 运行环境。
3. 所有 pi05 训练、归一化、LeRobot 转换命令都用 `./.venv/bin/python` 执行。
