#!/bin/bash

# Black Box Challenge - Baseline Implementation
# This script takes three parameters and outputs the reimbursement amount
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Run the Python implementation
python3 calculate_reimbursement.py "$1" "$2" "$3"