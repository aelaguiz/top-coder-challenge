#!/bin/bash

# Black Box Challenge - Version 2 Implementation
# This script takes three parameters and outputs the reimbursement amount
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Run the Python implementation (V2)
python3 calculate_reimbursement_v2.py "$1" "$2" "$3"