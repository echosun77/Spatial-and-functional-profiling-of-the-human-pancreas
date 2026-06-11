#!/bin/bash

# Get the current date in YYYY-MM-DD format
DATE=$(date +%Y-%m-%d_%H-%M-%S)

# Define log filename with the current date
LOG_FILE="logs/memento_cts_csts_by_sex_$DATE.log"

# Run the Python script and save logs with the date
python -u memento_cts_csts_by_sex_rV2.py 2>&1 | tee "$LOG_FILE"

