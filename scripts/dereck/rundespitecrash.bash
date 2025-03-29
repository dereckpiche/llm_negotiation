#!/bin/bash

# Path to your Python script
PYTHON_SCRIPT="src/run.py"

# Function to run the Python script
run_script() {
    python $PYTHON_SCRIPT
}

# Loop to keep running the script until it succeeds
while true; do
    run_script
    if [ $? -eq 0 ]; then
        echo "Script completed successfully."
        break
    else
        echo "Script crashed. Restarting..."
    fi
done