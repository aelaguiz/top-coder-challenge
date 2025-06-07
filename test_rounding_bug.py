#!/usr/bin/env python3
"""Test for systematic rounding bug with .49/.99 receipts."""

import json
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

print("## Testing Rounding Bug Hypothesis\n")

# Separate cases by receipt ending
cases_49_99 = []
cases_other = []

for case in cases:
    inp = case['input']
    receipts = inp['total_receipts_amount']
    cents = int(round((receipts % 1) * 100))
    
    case_data = {
        'days': inp['trip_duration_days'],
        'miles': inp['miles_traveled'],
        'receipts': receipts,
        'expected': case['expected_output'],
        'cents': cents
    }
    
    if cents in [49, 99]:
        cases_49_99.append(case_data)
    else:
        cases_other.append(case_data)

print(f"Cases with .49/.99: {len(cases_49_99)}")
print(f"Other cases: {len(cases_other)}\n")

# Build a simple model on non-.49/.99 cases
if len(cases_other) > 100:
    # Extract features
    X_other = [[c['days'], c['miles'], c['receipts']] for c in cases_other]
    y_other = [c['expected'] for c in cases_other]
    
    # Train model
    model = LinearRegression()
    model.fit(X_other, y_other)
    
    # Predict for .49/.99 cases
    print("### Prediction Analysis")
    print("\nChecking if .49/.99 cases get bonus/penalty:\n")
    
    total_diff = 0
    positive_diff = 0
    
    for case in cases_49_99:
        X = [[case['days'], case['miles'], case['receipts']]]
        predicted = model.predict(X)[0]
        actual = case['expected']
        diff = actual - predicted
        pct_diff = (diff / predicted * 100) if predicted > 0 else 0
        
        total_diff += diff
        if diff > 0:
            positive_diff += 1
        
        print(f"  {case['days']}d, {case['miles']:.0f}mi, ${case['receipts']:.2f}:")
        print(f"    Predicted: ${predicted:.2f}, Actual: ${actual:.2f}")
        print(f"    Difference: ${diff:.2f} ({pct_diff:+.1f}%)")
        if abs(diff) > 50:
            print(f"    *** SIGNIFICANT DIFFERENCE ***")
        print()
    
    avg_diff = total_diff / len(cases_49_99) if cases_49_99 else 0
    print(f"\nSummary:")
    print(f"  Average difference: ${avg_diff:.2f}")
    print(f"  Cases with positive difference: {positive_diff}/{len(cases_49_99)} ({positive_diff/len(cases_49_99)*100:.1f}%)")
    
    if avg_diff > 10:
        print(f"  \n*** EVIDENCE OF BONUS for .49/.99 receipts! ***")
    elif avg_diff < -10:
        print(f"  \n*** EVIDENCE OF PENALTY for .49/.99 receipts! ***")
    else:
        print(f"  \nNo clear systematic bias for .49/.99 receipts")

# Check if reimbursement amounts have rounding patterns
print("\n### Reimbursement Rounding Analysis")

# Check last two digits of reimbursement
last_two_digits = {}
for case in cases:
    reimb = case['expected_output']
    # Get last two digits (cents)
    cents = int(round((reimb % 1) * 100))
    last_two = cents
    
    if last_two not in last_two_digits:
        last_two_digits[last_two] = 0
    last_two_digits[last_two] += 1

# Find patterns
print("\nMost common reimbursement cent values:")
common = sorted(last_two_digits.items(), key=lambda x: x[1], reverse=True)[:15]
for cents, count in common:
    print(f"  .{cents:02d}: {count} cases")

# Check if certain endings are overrepresented
expected_per_ending = len(cases) / 100  # If uniform distribution
print(f"\nExpected cases per ending (uniform): {expected_per_ending:.1f}")
print("\nSignificantly overrepresented endings:")
for cents, count in last_two_digits.items():
    if count > expected_per_ending * 1.5:  # 50% more than expected
        print(f"  .{cents:02d}: {count} cases ({count/expected_per_ending:.1f}x expected)")

print("\n## Conclusions")
print("1. Check if .49/.99 receipt cases systematically get different treatment")
print("2. Look for non-uniform distribution in reimbursement cent values")
print("3. Consider if rounding rules create calculation artifacts")