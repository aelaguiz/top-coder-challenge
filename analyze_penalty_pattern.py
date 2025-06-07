#!/usr/bin/env python3
"""Analyze the .49/.99 penalty pattern to find the right formula."""

import json
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

# Get all .49/.99 cases with their predicted values
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

# Build model on non-.49/.99 cases to get "normal" predictions
X_other = [[c['days'], c['miles'], c['receipts']] for c in cases_other]
y_other = [c['expected'] for c in cases_other]
model = LinearRegression()
model.fit(X_other, y_other)

print("## Analyzing .49/.99 Penalty Pattern\n")

# Calculate penalty percentages
penalties = []
for case in cases_49_99:
    X = [[case['days'], case['miles'], case['receipts']]]
    predicted = model.predict(X)[0]
    actual = case['expected']
    
    penalty_pct = (predicted - actual) / predicted * 100 if predicted > 0 else 0
    penalties.append(penalty_pct)
    
    print(f"{case['days']}d, {case['miles']:.0f}mi, ${case['receipts']:.2f}:")
    print(f"  Normal prediction: ${predicted:.2f}")
    print(f"  Actual (with penalty): ${actual:.2f}")
    print(f"  Penalty: {penalty_pct:.1f}%\n")

# Statistics
print("\n### Penalty Statistics")
print(f"Average penalty: {np.mean(penalties):.1f}%")
print(f"Median penalty: {np.median(penalties):.1f}%")
print(f"Min penalty: {np.min(penalties):.1f}%")
print(f"Max penalty: {np.max(penalties):.1f}%")
print(f"Std deviation: {np.std(penalties):.1f}%")

# Try different penalty formulas
print("\n### Testing Penalty Formulas\n")

# Test fixed percentage penalties
for penalty_pct in [30, 35, 40, 45, 50, 55, 60]:
    errors = []
    for case in cases_49_99:
        X = [[case['days'], case['miles'], case['receipts']]]
        predicted = model.predict(X)[0]
        with_penalty = predicted * (1 - penalty_pct/100)
        error = abs(with_penalty - case['expected'])
        errors.append(error)
    
    avg_error = np.mean(errors)
    print(f"Fixed {penalty_pct}% penalty: avg error ${avg_error:.2f}")

# Test variable penalty based on receipt amount
print("\n### Variable Penalty by Receipt Amount")

# Group by receipt ranges
ranges = [
    (0, 100, 'Low'),
    (100, 500, 'Medium'),
    (500, 1000, 'Medium-High'),
    (1000, 3000, 'High')
]

for low, high, label in ranges:
    range_cases = [c for c in cases_49_99 if low <= c['receipts'] < high]
    if range_cases:
        range_penalties = []
        for case in range_cases:
            X = [[case['days'], case['miles'], case['receipts']]]
            predicted = model.predict(X)[0]
            penalty_pct = (predicted - case['expected']) / predicted * 100 if predicted > 0 else 0
            range_penalties.append(penalty_pct)
        
        if range_penalties:
            avg = np.mean(range_penalties)
            print(f"{label} receipts (${low}-${high}): avg penalty {avg:.1f}% ({len(range_cases)} cases)")

print("\n### Recommendation")
print("The penalty varies significantly (12-78%) but a fixed 45-50% penalty")
print("would be a reasonable approximation for most cases.")