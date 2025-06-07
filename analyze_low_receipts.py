#!/usr/bin/env python3
"""Analyze low receipt cases to find better formulas."""

import json
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load data
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

# Extract low receipt cases
low_receipt_cases = []
for case in cases:
    inp = case['input']
    if inp['total_receipts_amount'] < 30:
        low_receipt_cases.append({
            'days': inp['trip_duration_days'],
            'miles': inp['miles_traveled'],
            'receipts': inp['total_receipts_amount'],
            'expected': case['expected_output']
        })

print(f"Found {len(low_receipt_cases)} low receipt cases (<$30)\n")

# Group by trip length
by_trip_length = {
    'single_day': [],
    'short_trip': [],  # 2-3 days
    'medium_trip': [],  # 4-6 days
    'long_trip': []  # 7+ days
}

for case in low_receipt_cases:
    if case['days'] == 1:
        by_trip_length['single_day'].append(case)
    elif case['days'] <= 3:
        by_trip_length['short_trip'].append(case)
    elif case['days'] <= 6:
        by_trip_length['medium_trip'].append(case)
    else:
        by_trip_length['long_trip'].append(case)

# Analyze each group
print("## Analysis by Trip Length\n")

for group_name, cases in by_trip_length.items():
    if not cases:
        continue
        
    print(f"### {group_name.replace('_', ' ').title()} ({len(cases)} cases)")
    
    # Try different formulas
    formulas = [
        ('$100/day + $0.75/mile', lambda c: 100 * c['days'] + 0.75 * c['miles']),
        ('$100/day + $0.50/mile', lambda c: 100 * c['days'] + 0.50 * c['miles']),
        ('$100/day + $0.40/mile', lambda c: 100 * c['days'] + 0.40 * c['miles']),
        ('$80/day + $0.75/mile', lambda c: 80 * c['days'] + 0.75 * c['miles']),
        ('$80/day + $0.50/mile', lambda c: 80 * c['days'] + 0.50 * c['miles']),
        ('Linear fit', None)  # Will fit linear regression
    ]
    
    # Test each formula
    results = []
    
    for formula_name, formula_func in formulas:
        if formula_func:
            errors = []
            for case in cases:
                predicted = formula_func(case)
                error = abs(predicted - case['expected'])
                errors.append(error)
            avg_error = np.mean(errors)
            results.append((formula_name, avg_error))
        else:
            # Linear regression
            X = [[c['days'], c['miles']] for c in cases]
            y = [c['expected'] for c in cases]
            
            if len(cases) >= 3:  # Need at least 3 points for meaningful regression
                lr = LinearRegression()
                lr.fit(X, y)
                
                errors = []
                for case in cases:
                    predicted = lr.predict([[case['days'], case['miles']]])[0]
                    error = abs(predicted - case['expected'])
                    errors.append(error)
                avg_error = np.mean(errors)
                
                formula_desc = f"${lr.coef_[0]:.2f}/day + ${lr.coef_[1]:.4f}/mile + ${lr.intercept_:.2f}"
                results.append((formula_desc, avg_error))
    
    # Sort by error
    results.sort(key=lambda x: x[1])
    
    print("\nFormula comparison (avg error):")
    for formula, error in results:
        print(f"  {formula}: ${error:.2f}")
    
    # Show example cases
    print("\nExample cases:")
    for i, case in enumerate(cases[:3]):
        print(f"  {case['days']}d, {case['miles']:.0f}mi, ${case['receipts']:.2f} → ${case['expected']:.2f}")
    
    print()

# Analyze the pattern more deeply
print("\n## Deeper Pattern Analysis\n")

# Check if there's a base amount plus per-mile rate that varies by trip length
for group_name, cases in by_trip_length.items():
    if not cases or len(cases) < 3:
        continue
    
    print(f"### {group_name.replace('_', ' ').title()}")
    
    # Calculate implicit per-mile rate for each case
    # Assuming: reimbursement = base_per_day * days + rate_per_mile * miles
    # Try different base rates
    for base_per_day in [80, 90, 100, 110]:
        rates = []
        for case in cases:
            # reimbursement = base_per_day * days + rate_per_mile * miles
            # rate_per_mile = (reimbursement - base_per_day * days) / miles
            if case['miles'] > 0:
                rate = (case['expected'] - base_per_day * case['days']) / case['miles']
                rates.append(rate)
        
        if rates:
            avg_rate = np.mean(rates)
            std_rate = np.std(rates)
            print(f"  Base ${base_per_day}/day → avg mile rate: ${avg_rate:.3f} (std: ${std_rate:.3f})")

print("\n## Recommendation\n")
print("Based on the analysis, implement different formulas by trip length:")
print("- Single day trips: Use fitted formula from linear regression")
print("- Multi-day trips: Consider using ML model or more complex rules")
print("- The simple linear formula breaks down for longer trips")