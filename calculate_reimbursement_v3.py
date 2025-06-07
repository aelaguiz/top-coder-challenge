#!/usr/bin/env python3

import sys
import joblib
import numpy as np
import json
import logging

# Configure logging to write to disk
logging.basicConfig(
    filename='calculate_reimbursement.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load expected values for logging/debugging
with open('public_cases.json', 'r') as f:
    PUBLIC_CASES = json.load(f)
    EXPECTED_OUTPUTS = {
        (case['input']['trip_duration_days'], 
         case['input']['miles_traveled'], 
         case['input']['total_receipts_amount']): case['expected_output']
        for case in PUBLIC_CASES
    }

# Load the trained model
model = joblib.load('model.pkl')

# Parse arguments
days = int(sys.argv[1])
miles = float(sys.argv[2])  # Can be float!
receipts = float(sys.argv[3])

logging.info(f"\nInput: days={days}, miles={miles}, receipts={receipts}")

# Check if this is a low receipt case
if receipts < 30:
    # Use trip-length-specific formulas based on our analysis
    if days == 1:
        # Single day: $106.12 + $0.525/mile (avg error $11.24)
        prediction = 106.12 + 0.525 * miles
        logging.info(f"Using single day formula: $106.12 + $0.525*{miles} = ${prediction:.2f}")
    elif days <= 3:
        # Short trip (2-3 days): $112.02/day + $0.857/mile - $40.10 (avg error $17.11)
        prediction = 112.02 * days + 0.857 * miles - 40.10
        logging.info(f"Using short trip formula: $112.02*{days} + $0.857*{miles} - $40.10 = ${prediction:.2f}")
    elif days <= 6:
        # Medium trip (4-6 days): $100/day + $0.50/mile (avg error $11.62)
        prediction = 100 * days + 0.50 * miles
        logging.info(f"Using medium trip formula: $100*{days} + $0.50*{miles} = ${prediction:.2f}")
    else:
        # Long trip (7+ days): Complex pattern, use ML model instead
        # The linear regression shows negative mile rates which doesn't make sense
        # Fall back to ML model for these cases
        features = np.array([[days, miles, receipts]])
        prediction = model.predict(features)[0]
        logging.info(f"Using ML model for long trip with low receipts: ${prediction:.2f}")
else:
    # For receipts >= $30, use the ML model
    features = np.array([[days, miles, receipts]])
    prediction = model.predict(features)[0]
    logging.info(f"Using ML model for receipts >= $30: ${prediction:.2f}")

# Look up expected value if available
expected_key = (days, miles, receipts)
if expected_key in EXPECTED_OUTPUTS:
    expected = EXPECTED_OUTPUTS[expected_key]
    error = abs(prediction - expected)
    logging.info(f"Expected: ${expected:.2f}, Predicted: ${prediction:.2f}, Error: ${error:.2f}")
else:
    logging.info(f"No expected value found for this input")

# Output the result
print(f"{prediction:.2f}")