# Multi-agent Reinforcement Learning with Large Language Models

## Contributors
- Dereck Piche (https://dereckpiche.github.io/)
- Muqeeth Mohammed
- Uday Kapoor

<img src="logo.png" alt="My Image" style="display: block; margin: 0 auto; width: 400px;">



## Installation

It is recommended to use python version `3.11` and `CUDA 12.4`.

```bash
pip install -r requirements.txt
pip install flash_attn --no-build-isolation
cd path/to/repo
pip install -e .
```

and run

```python
import sys
sys.path.append('your-path-to-the-repo')
```
in order to add the repository to your system path.

To run with OpenAI API models, set
```bash
export OPENAI_API_KEY=your/api/key
```

## Development

```bash
pip install pre-commit
pre-commit install
pip install nbstripout
nbstripout --install
```

## Running Experiments

In order to launch a policy gradient training loop, use
```bash
python run.py --config-name your-config
```

To add render files to your output folder, use
```bash
python render.py --simulation_name path
```
Example: `python render.py --nego tas_rps_ad_align_coop_push_32_games_beta_3`.
