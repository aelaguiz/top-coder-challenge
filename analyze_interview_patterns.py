#!/usr/bin/env python3
"""Analyze specific patterns mentioned in interviews."""

import json
import numpy as np
from collections import defaultdict

# Load data
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

print("## Analyzing Interview Claims\n")

# 1. Check for $847 "magic number" (Marcus)
print("### 1. Magic Number $847 (Marcus)")
for case in cases:
    reimb = case['expected_output']
    if 840 <= reimb <= 850:
        inp = case['input']
        print(f"  Found: ${reimb:.2f} - {inp['trip_duration_days']}d, {inp['miles_traveled']}mi, ${inp['total_receipts_amount']:.2f}")

# 2. Check efficiency sweet spot 180-220 miles/day (Kevin)
print("\n### 2. Efficiency Sweet Spot 180-220 mpd (Kevin)")
sweet_spot_cases = []
other_efficiency_cases = []

for case in cases:
    inp = case['input']
    mpd = inp['miles_traveled'] / inp['trip_duration_days']
    
    if 180 <= mpd <= 220:
        sweet_spot_cases.append((mpd, case['expected_output'], inp))
    elif 150 <= mpd <= 250:  # Similar range for comparison
        other_efficiency_cases.append((mpd, case['expected_output'], inp))

if sweet_spot_cases:
    avg_sweet = np.mean([r for _, r, _ in sweet_spot_cases])
    avg_other = np.mean([r for _, r, _ in other_efficiency_cases])
    print(f"  Sweet spot (180-220 mpd): {len(sweet_spot_cases)} cases, avg reimb: ${avg_sweet:.2f}")
    print(f"  Nearby range: {len(other_efficiency_cases)} cases, avg reimb: ${avg_other:.2f}")
    print(f"  Difference: ${avg_sweet - avg_other:.2f}")

# 3. Check rounding bug for .49/.99 receipts (Lisa)
print("\n### 3. Rounding Bug .49/.99 (Lisa)")
rounding_cases = defaultdict(list)

for case in cases:
    inp = case['input']
    receipts = inp['total_receipts_amount']
    cents = int(round((receipts % 1) * 100))
    
    if cents in [49, 99]:
        rounding_cases['special'].append(case['expected_output'])
    else:
        rounding_cases['normal'].append(case['expected_output'])

if rounding_cases['special']:
    avg_special = np.mean(rounding_cases['special'])
    avg_normal = np.mean(rounding_cases['normal'])
    print(f"  .49/.99 endings: {len(rounding_cases['special'])} cases, avg: ${avg_special:.2f}")
    print(f"  Other endings: {len(rounding_cases['normal'])} cases, avg: ${avg_normal:.2f}")
    print(f"  Difference: ${avg_special - avg_normal:.2f}")
    
    # Check if they're systematically rounded up
    print("\n  Examples of .49/.99 cases:")
    for case in cases[:5]:
        inp = case['input']
        receipts = inp['total_receipts_amount']
        cents = int(round((receipts % 1) * 100))
        if cents in [49, 99]:
            print(f"    ${receipts:.2f} → ${case['expected_output']:.2f}")

# 4. 5-day trips bonus (Lisa)
print("\n### 4. Five-Day Bonus (Lisa)")
by_days = defaultdict(list)

for case in cases:
    inp = case['input']
    days = inp['trip_duration_days']
    by_days[days].append(case['expected_output'])

for days in [4, 5, 6]:
    if by_days[days]:
        avg = np.mean(by_days[days])
        print(f"  {days}-day trips: {len(by_days[days])} cases, avg: ${avg:.2f}")

# 5. Check spending per day patterns (Kevin)
print("\n### 5. Optimal Spending Ranges (Kevin)")
print("  Kevin's claims:")
print("  - Short trips (1-3d): < $75/day")
print("  - Medium trips (4-6d): < $120/day")
print("  - Long trips (7+d): < $90/day")

# Analyze actual patterns
spending_analysis = {
    'short': {'good': [], 'bad': []},
    'medium': {'good': [], 'bad': []},
    'long': {'good': [], 'bad': []}
}

for case in cases:
    inp = case['input']
    days = inp['trip_duration_days']
    receipts = inp['total_receipts_amount']
    spd = receipts / days if days > 0 else 0
    reimb = case['expected_output']
    
    if days <= 3:
        if spd < 75:
            spending_analysis['short']['good'].append(reimb)
        else:
            spending_analysis['short']['bad'].append(reimb)
    elif days <= 6:
        if spd < 120:
            spending_analysis['medium']['good'].append(reimb)
        else:
            spending_analysis['medium']['bad'].append(reimb)
    else:
        if spd < 90:
            spending_analysis['long']['good'].append(reimb)
        else:
            spending_analysis['long']['bad'].append(reimb)

print("\n  Analysis results:")
for trip_type, data in spending_analysis.items():
    if data['good'] and data['bad']:
        avg_good = np.mean(data['good'])
        avg_bad = np.mean(data['bad'])
        print(f"  {trip_type.title()} trips:")
        print(f"    Within Kevin's range: {len(data['good'])} cases, avg: ${avg_good:.2f}")
        print(f"    Outside range: {len(data['bad'])} cases, avg: ${avg_bad:.2f}")
        print(f"    Difference: ${avg_good - avg_bad:.2f}")

# 6. Check for mileage tiers/drops after 100 miles (Lisa)
print("\n### 6. Mileage Tiers (Lisa)")
print("  Lisa claims rate drops after 100 miles...")

# Group by mileage ranges
mileage_ranges = [
    (0, 50, '0-50'),
    (50, 100, '50-100'),
    (100, 200, '100-200'),
    (200, 400, '200-400'),
    (400, 600, '400-600'),
    (600, 1000, '600-1000'),
    (1000, 2000, '1000+')
]

mileage_analysis = defaultdict(list)

for case in cases:
    inp = case['input']
    miles = inp['miles_traveled']
    reimb = case['expected_output']
    
    for low, high, label in mileage_ranges:
        if low <= miles < high:
            mileage_analysis[label].append(reimb)
            break

print("\n  Average reimbursement by mileage range:")
for _, _, label in mileage_ranges:
    if mileage_analysis[label]:
        avg = np.mean(mileage_analysis[label])
        count = len(mileage_analysis[label])
        print(f"  {label} miles: {count} cases, avg: ${avg:.2f}")

print("\n## Summary of Findings")
print("1. Magic $847: Found some cases in that range")
print("2. Efficiency bonus 180-220 mpd: Needs deeper analysis")
print("3. Rounding bug: Check if .49/.99 cases have patterns")
print("4. 5-day bonus: Already tested and rejected in hypothesis testing")
print("5. Spending ranges: Some evidence supporting Kevin's claims")
print("6. Mileage tiers: Clear progression showing increasing reimbursement")