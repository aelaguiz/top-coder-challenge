#!/usr/bin/env python3
"""Quick analysis of error patterns from the log."""

import re

# Extract high error cases from log
log_data = """
days=3, miles=93, receipts=1.42: Predicted: $221.44, Expected: $364.51
days=2, miles=13, receipts=4.67: Predicted: $125.44, Expected: $203.52
days=3, miles=88, receipts=5.78: Predicted: $227.47, Expected: $380.37
days=1, miles=76, receipts=13.74: Predicted: $94.55, Expected: $158.35
days=3, miles=41, receipts=4.52: Predicted: $187.39, Expected: $320.12
days=1, miles=140, receipts=22.71: Predicted: $115.57, Expected: $199.68
days=3, miles=121, receipts=21.17: Predicted: $250.83, Expected: $464.07
days=3, miles=117, receipts=21.99: Predicted: $222.42, Expected: $359.10
days=2, miles=202, receipts=21.24: Predicted: $214.67, Expected: $356.17
days=3, miles=80, receipts=21.05: Predicted: $220.91, Expected: $366.87
days=2, miles=21, receipts=20.04: Predicted: $125.36, Expected: $204.58
"""

# Parse and analyze
for line in log_data.strip().split('\n'):
    if 'days=' in line:
        parts = line.split(': ')
        inputs = parts[0]
        
        # Extract values
        days = int(re.search(r'days=(\d+)', inputs).group(1))
        miles = int(re.search(r'miles=(\d+)', inputs).group(1))
        receipts = float(re.search(r'receipts=([\d.]+)', inputs).group(1))
        expected = float(re.search(r'Expected: \$([\d.]+)', parts[1]).group(1))
        
        # Calculate potential formulas
        per_day = expected / days
        per_mile = expected / miles if miles > 0 else 0
        
        print(f"Days: {days}, Miles: {miles}, Receipts: ${receipts:.2f}")
        print(f"  Expected: ${expected:.2f}")
        print(f"  Per day: ${per_day:.2f}")
        print(f"  Per mile: ${per_mile:.2f}")
        
        # Test simple formula: $100/day + $0.50/mile
        simple_formula = 100 * days + 0.5 * miles
        print(f"  $100/day + $0.50/mile = ${simple_formula:.2f}")
        
        # Test another formula: $125/day + $0.40/mile
        formula2 = 125 * days + 0.4 * miles
        print(f"  $125/day + $0.40/mile = ${formula2:.2f}")
        
        print()