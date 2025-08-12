# Multi-agent Reinforcement Learning with Large Language Models

## Contributors
- Dereck Piche (https://dereckpiche.github.io/)
- Muqeeth Mohammed
- Uday Kapoor

<img src="logo.png" alt="My Image" style="display: block; margin: 0 auto; width: 400px;">



## Installation

It is recommended to use python version `3.11` and `CUDA 12.4`.

```bash
module unload python
module load anaconda/3
conda create -n venv python=3.11
conda activate venv
module load cudatoolkit/12.4
pip install torch==2.7.1
pip install psutil
pip install flash_attn --no-build-isolation
pip install "sglang[all]>=0.4.10.post2"
pip install -r requirements.txt
cd path/to/repo
pip install -e .
```


## Development

```bash
pip install pre-commit
pre-commit install
pip install nbstripout
nbstripout --install
```

## Running Experiments

```bash
python run.py --config-name your-config
```

To render
```bash
python scripts/basic_render.py --global_folder your-experiment-seed-xxx
```
