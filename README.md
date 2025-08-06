# Multi-agent Reinforcement Learning with Large Language Models

## Contributors
- Dereck Piche (https://dereckpiche.github.io/)
- Muqeeth Mohammed
- Uday Kapoor

<img src="logo.png" alt="My Image" style="display: block; margin: 0 auto; width: 400px;">





## Installation

It is recommended to use python version `3.10.11` and `CUDA 12.6`.

```bash
pip install --upgrade pip
pip install uv
uv pip install torch==2.6
uv pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/
uv pip install -r requirements.txt
uv pip install "sglang[all]>=0.4.9.post2"
cd path/to/repo
pip install -e .
```



## Development

```bash
uv pip install pre-commit
pre-commit install
```
