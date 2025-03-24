#!/bin/bash

# === Shared SLURM settings ===
PARTITION="long"
MEM="32G"
GPUS="a100l"
TIME="48:00:00"
CPUS="4"
ENV_NAME="/home/mila/d/dereck.piche/llm_negotiation/.venv/bin/activate"
LOGDIR="/home/mila/d/dereck.piche/aa_logs"
SCRIPT_PATH="/home/mila/d/dereck.piche/llm_negotiation/src/run.py"

mkdir -p "$LOGDIR"

# === Commands to run (varying only the Python command) ===
COMMANDS=(

  "python3 $SCRIPT_PATH experiment.base_seed=0"

  "python3 $SCRIPT_PATH experiment.base_seed=500"

  "python3 $SCRIPT_PATH experiment.base_seed=1000"

  "python3 $SCRIPT_PATH experiment.base_seed=1500"
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

module load python/3.10
module load cuda/11.7

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

done
