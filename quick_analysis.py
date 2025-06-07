#!/usr/bin/env python3
"""Quick analysis of low receipt cases to find pattern."""

import json

# Load public cases
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

# Analyze low receipt cases
print("Low Receipt Cases (<$30):")
print("-" * 80)

low_receipt_cases = []
for case in cases:
    inp = case['input']
    days = inp['trip_duration_days']
    miles = inp['miles_traveled']
    receipts = inp['total_receipts_amount']
    expected = case['expected_output']
    
    if receipts < 30:
        low_receipt_cases.append((days, miles, receipts, expected))
        
        # Test different formulas
        per_day = expected / days
        per_mile = expected / miles if miles > 0 else 0
        
        # Test formula: base_per_day + miles * rate
        # Try to find the base and rate
        base_100_rate = (expected - 100 * days) / miles if miles > 0 else 0
        base_125_rate = (expected - 125 * days) / miles if miles > 0 else 0
        
        print(f"Days: {days}, Miles: {miles}, Receipts: ${receipts:.2f} => ${expected:.2f}")
        print(f"  Per day: ${per_day:.2f}")
        print(f"  If $100/day base: ${base_100_rate:.3f}/mile")
        print(f"  If $125/day base: ${base_125_rate:.3f}/mile")
        
# Let's look for a pattern
print("\n\nTesting formula: $100/day + $variable/mile")
print("-" * 80)

# Group by days to see if rate changes
from collections import defaultdict
by_days = defaultdict(list)

for days, miles, receipts, expected in low_receipt_cases:
    if miles > 0:
        rate = (expected - 100 * days) / miles
        by_days[days].append((miles, receipts, expected, rate))

for days in sorted(by_days.keys()):
    rates = [r[3] for r in by_days[days]]
    if rates:
        avg_rate = sum(rates) / len(rates)
        print(f"{days} days: avg rate = ${avg_rate:.3f}/mile (n={len(rates)})")