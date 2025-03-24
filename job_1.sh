#!/bin/bash
#SBATCH --job-name=aa_job_1
#SBATCH --output=/home/mila/d/dereck.piche/aa_logs/aa_job_1_%j.out
#SBATCH --error=/home/mila/d/dereck.piche/aa_logs/aa_job_1_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=long
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@60

module load python/3.10
module load cuda/11.7

source activate /home/mila/d/dereck.piche/llm_negotiation/.venv/bin/activate

echo "Starting job aa_job_1 on $(hostname)"

while true; do
  echo "Running command: python3 /home/mila/d/dereck.piche/llm_negotiation/src/run.py experiment.base_seed=500"
  python3 /home/mila/d/dereck.piche/llm_negotiation/src/run.py experiment.base_seed=500
  EXIT_CODE=$?
  if [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully with exit code $EXIT_CODE."
    break
  else
    echo "Command failed with exit code $EXIT_CODE. Relaunching in 10 seconds..."
    sleep 10
  fi
done
