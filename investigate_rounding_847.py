#!/usr/bin/env python3
"""Deep dive into rounding bug and $847 pattern."""

import json
import numpy as np
from collections import defaultdict

# Load data
with open('public_cases.json', 'r') as f:
    cases = json.load(f)

print("## Deep Investigation\n")

# 1. Investigate .49/.99 pattern more carefully
print("### 1. Rounding Pattern Analysis (.49/.99)")

# Group by receipt ending and check if they're mostly low receipts
by_ending = defaultdict(list)

for case in cases:
    inp = case['input']
    receipts = inp['total_receipts_amount']
    cents = int(round((receipts % 1) * 100))
    
    by_ending[cents].append({
        'receipts': receipts,
        'reimb': case['expected_output'],
        'days': inp['trip_duration_days'],
        'miles': inp['miles_traveled']
    })

# Check if .49/.99 are mostly low receipts
print("\nReceipt amounts for .49/.99 endings:")
for ending in [49, 99]:
    if by_ending[ending]:
        receipt_amounts = [c['receipts'] for c in by_ending[ending]]
        print(f"\n.{ending} endings ({len(receipt_amounts)} cases):")
        print(f"  Min receipt: ${min(receipt_amounts):.2f}")
        print(f"  Max receipt: ${max(receipt_amounts):.2f}")
        print(f"  Avg receipt: ${np.mean(receipt_amounts):.2f}")
        
        # Show some examples
        print("  Examples:")
        for c in sorted(by_ending[ending], key=lambda x: x['receipts'])[:5]:
            print(f"    ${c['receipts']:.2f} → ${c['reimb']:.2f} ({c['days']}d, {c['miles']}mi)")

# 2. Check if reimbursement amounts ending in specific cents are common
print("\n### 2. Reimbursement Amount Endings")
reimb_endings = defaultdict(int)

for case in cases:
    reimb = case['expected_output']
    cents = int(round((reimb % 1) * 100))
    reimb_endings[cents] += 1

# Find most common endings
common_endings = sorted(reimb_endings.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nMost common reimbursement endings:")
for cents, count in common_endings:
    print(f"  .{cents:02d}: {count} cases")

# 3. Investigate $847 pattern more deeply
print("\n### 3. $847 Pattern Investigation")

# Look at cases near $847
print("\nCases near $847 (±$10):")
near_847 = []
for case in cases:
    reimb = case['expected_output']
    if 837 <= reimb <= 857:
        inp = case['input']
        near_847.append({
            'reimb': reimb,
            'days': inp['trip_duration_days'],
            'miles': inp['miles_traveled'],
            'receipts': inp['total_receipts_amount'],
            'mpd': inp['miles_traveled'] / inp['trip_duration_days'],
            'rpd': inp['total_receipts_amount'] / inp['trip_duration_days']
        })

for c in sorted(near_847, key=lambda x: x['reimb']):
    print(f"  ${c['reimb']:.2f}: {c['days']}d, {c['miles']:.0f}mi, ${c['receipts']:.2f} (mpd: {c['mpd']:.1f}, rpd: ${c['rpd']:.1f})")

# Look for patterns
if near_847:
    print(f"\n  Analysis of {len(near_847)} cases near $847:")
    print(f"  Avg days: {np.mean([c['days'] for c in near_847]):.1f}")
    print(f"  Avg miles: {np.mean([c['miles'] for c in near_847]):.1f}")
    print(f"  Avg receipts: ${np.mean([c['receipts'] for c in near_847]):.2f}")
    print(f"  Avg mpd: {np.mean([c['mpd'] for c in near_847]):.1f}")

# 4. Check for specific reimbursement values that appear multiple times
print("\n### 4. Repeated Reimbursement Values")
reimb_counts = defaultdict(int)
reimb_examples = defaultdict(list)

for case in cases:
    reimb = case['expected_output']
    reimb_counts[reimb] += 1
    if reimb_counts[reimb] <= 3:  # Keep first 3 examples
        inp = case['input']
        reimb_examples[reimb].append(f"{inp['trip_duration_days']}d, {inp['miles_traveled']}mi, ${inp['total_receipts_amount']:.2f}")

# Find reimbursements that appear multiple times
repeated = [(reimb, count) for reimb, count in reimb_counts.items() if count > 1]
repeated.sort(key=lambda x: x[1], reverse=True)

print("\nReimbursement values appearing multiple times:")
for reimb, count in repeated[:10]:
    print(f"  ${reimb:.2f}: {count} times")
    for ex in reimb_examples[reimb]:
        print(f"    - {ex}")

print("\n## Key Insights")
print("1. Check if .49/.99 receipts are mostly low amounts (penalty correlation)")
print("2. Look for systematic rounding in reimbursement amounts")
print("3. The $847 cluster might indicate a calculation ceiling or special rule")
print("4. Repeated exact values suggest discrete calculation outcomes")