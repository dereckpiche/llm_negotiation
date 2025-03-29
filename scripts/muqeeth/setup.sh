export MM_ROOT=`pwd`
export PYTHONPATH=$MM_ROOT:$PYTHONPATH
export PYTHON_EXEC=python
mkdir -p ~/.cache/llm_negotiation/
export HUGGINGFACE_HUB_CACHE=~/.cache/llm_negotiation/
export HF_HOME=~/.cache/llm_negotiation/
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=True
export WANDB_PROJECT=llm_negotiation
export DATA_CACHE=/network/scratch/m/mohammed.muqeeth/llm_negotiation/datasets_offline
export EXP_OUT=/network/scratch/m/mohammed.muqeeth/llm_negotiation/exp_out
export HYDRA_FULL_ERROR=1