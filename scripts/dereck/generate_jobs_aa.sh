#!/bin/bash

# === Shared SLURM settings ===
PARTITION="long"
MEM="48G"
GPUS="a100l"
TIME="48:00:00"
CPUS="6"
ENV_NAME="/home/mila/d/dereck.piche/llm_negotiation/.venv/bin/activate"
LOGDIR="/home/mila/d/dereck.piche/llm_negotiation/slurm_logs"
SCRIPT_PATH="/home/mila/d/dereck.piche/llm_negotiation/src/run.py"

mkdir -p "$LOGDIR"

# === Commands to run (varying only the Python command) ===
# COMMANDS=(

#   "python3 $SCRIPT_PATH --config-name=adv_align experiment.base_seed=1 hydra.run.dir=/home/mila/d/dereck.piche/scratch/aa_regulated training.agents.alice.training_data_func_args.score_method_kwargs.regulate_var=True training.agents.bob.training_data_func_args.score_method_kwargs.regulate_var=True"

#   "python3 $SCRIPT_PATH --config-name=adv_align experiment.base_seed=2 hydra.run.dir=/home/mila/d/dereck.piche/scratch/aa_tiny_beta training.agents.alice.training_data_func_args.score_method_kwargs.beta=0.25 training.agents.bob.training_data_func_args.score_method_kwargs.beta=0.25"

#   "python3 $SCRIPT_PATH --config-name=adv_align experiment.base_seed=3 hydra.run.dir=/home/mila/d/dereck.piche/scratch/aa_small_beta training.agents.alice.training_data_func_args.score_method_kwargs.beta=0.5 training.agents.bob.training_data_func_args.score_method_kwargs.beta=0.5"

#   "python3 $SCRIPT_PATH --config-name=adv_align experiment.base_seed=4 hydra.run.dir=/home/mila/d/dereck.piche/scratch/aa_medium_beta training.agents.alice.training_data_func_args.score_method_kwargs.beta=0.8 training.agents.bob.training_data_func_args.score_method_kwargs.beta=0.8"

#   "python3 $SCRIPT_PATH --config-name=adv_align experiment.base_seed=5 hydra.run.dir=/home/mila/d/dereck.piche/scratch/aa_standard_beta training.agents.alice.training_data_func_args.score_method_kwargs.beta=1.0 training.agents.bob.training_data_func_args.score_method_kwargs.beta=1.0"

#   "python3 $SCRIPT_PATH --config-name=adv_align experiment.base_seed=6 hydra.run.dir=/home/mila/d/dereck.piche/scratch/aa_big_beta training.agents.alice.training_data_func_args.score_method_kwargs.beta=2.0 training.agents.bob.training_data_func_args.score_method_kwargs.beta=2.0"
# )

COMMANDS=(
  "python $SCRIPT_PATH --config-name=dond --config-path=/home/mila/d/dereck.piche/llm_negotiation/scripts/dereck --config-name=dond_aa_version experiment.base_seed=1"
  "python $SCRIPT_PATH --config-name=dond --config-path=/home/mila/d/dereck.piche/llm_negotiation/scripts/dereck --config-name=dond_aa_version experiment.base_seed=53"
  "python $SCRIPT_PATH --config-name=dond --config-path=/home/mila/d/dereck.piche/llm_negotiation/scripts/dereck --config-name=dond_aa_version experiment.base_seed=97"
  "python $SCRIPT_PATH --config-name=dond --config-path=/home/mila/d/dereck.piche/llm_negotiation/scripts/dereck --config-name=dond_aa_version experiment.base_seed=157"
  "python $SCRIPT_PATH --config-name=dond --config-path=/home/mila/d/dereck.piche/llm_negotiation/scripts/dereck --config-name=dond_aa_version experiment.base_seed=468"
)


# === Loop to create job scripts ===
for i in "${!COMMANDS[@]}"; do
  CMD="${COMMANDS[$i]}"
  JOB_SCRIPT="job_$i.sh"
  JOB_NAME="aa_job_$i"

  cat <<EOT > $JOB_SCRIPT
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$LOGDIR/${JOB_NAME}_%j.out
#SBATCH --error=$LOGDIR/${JOB_NAME}_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:$GPUS
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --partition=$PARTITION
#SBATCH --time=$TIME
#SBATCH --requeue
#SBATCH --signal=B:USR1@60


source activate $ENV_NAME

echo "Starting job $JOB_NAME on \$(hostname)"

while true; do
  echo "Running command: ${CMD}"
  ${CMD}
  EXIT_CODE=\$?
  if [ \$EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully with exit code \$EXIT_CODE."
    break
  else
    echo "Command failed with exit code \$EXIT_CODE. Relaunching in 10 seconds..."
    sleep 10
  fi
done
EOT

  # Removed the sbatch command to prevent job submission
  # sbatch "$JOB_SCRIPT"
done
