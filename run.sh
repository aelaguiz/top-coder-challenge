#!/bin/bash

# Black Box Challenge - V4 Implementation (with all improvements)
# This script takes three parameters and outputs the reimbursement amount
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Run the Python implementation (V4 with low receipt formulas and .49/.99 penalty)
python3 calculate_reimbursement.py "$1" "$2" "$3"