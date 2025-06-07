#!/usr/bin/env python3
"""Check if .49/.99 cases improved with the penalty."""

import json

# Load V2 results (before penalty)
with open('parallel_eval_results_backup.json', 'r') as f:
    v2_data = json.load(f)

# Load V4 results (with penalty)
with open('parallel_eval_results.json', 'r') as f:
    v4_data = json.load(f)

# Extract results into dict for easy comparison
v2_results = {}
v4_results = {}

for r in v2_data['results']:
    key = (r['days'], r['miles'], r['receipts'])
    v2_results[key] = r

for r in v4_data['results']:
    key = (r['days'], r['miles'], r['receipts'])
    v4_results[key] = r

print("## .49/.99 Cases Comparison (V2 vs V4)\n")

# Find .49/.99 cases
cases_49_99 = []

for key, v4_result in v4_results.items():
    receipts = key[2]
    cents = int(round((receipts % 1) * 100))
    
    if cents in [49, 99]:
        v2_result = v2_results.get(key)
        if v2_result:
            improvement = v2_result['error'] - v4_result['error']
            cases_49_99.append({
                'key': key,
                'v2_error': v2_result['error'],
                'v4_error': v4_result['error'],
                'improvement': improvement,
                'v2_actual': v2_result['actual'],
                'v4_actual': v4_result['actual'],
                'expected': v4_result['expected']
            })

# Sort by improvement
cases_49_99.sort(key=lambda x: x['improvement'], reverse=True)

print(f"Found {len(cases_49_99)} cases with .49/.99 receipts\n")

print("Top improvements:")
for case in cases_49_99[:10]:
    days, miles, receipts = case['key']
    print(f"\n{days}d, {miles:.0f}mi, ${receipts:.2f}:")
    print(f"  Expected: ${case['expected']:.2f}")
    print(f"  V2 prediction: ${case['v2_actual']:.2f} (error: ${case['v2_error']:.2f})")
    print(f"  V4 prediction: ${case['v4_actual']:.2f} (error: ${case['v4_error']:.2f})")
    print(f"  Improvement: ${case['improvement']:.2f}")

print("\n\nWorst cases (negative improvement):")
for case in cases_49_99[-5:]:
    days, miles, receipts = case['key']
    print(f"\n{days}d, {miles:.0f}mi, ${receipts:.2f}:")
    print(f"  Expected: ${case['expected']:.2f}")
    print(f"  V2 prediction: ${case['v2_actual']:.2f} (error: ${case['v2_error']:.2f})")
    print(f"  V4 prediction: ${case['v4_actual']:.2f} (error: ${case['v4_error']:.2f})")
    print(f"  Worsened by: ${-case['improvement']:.2f}")

# Summary statistics
total_v2_error = sum(c['v2_error'] for c in cases_49_99)
total_v4_error = sum(c['v4_error'] for c in cases_49_99)
avg_v2_error = total_v2_error / len(cases_49_99) if cases_49_99 else 0
avg_v4_error = total_v4_error / len(cases_49_99) if cases_49_99 else 0

print(f"\n\n### Summary for .49/.99 cases:")
print(f"Average V2 error: ${avg_v2_error:.2f}")
print(f"Average V4 error: ${avg_v4_error:.2f}")
print(f"Overall improvement: ${avg_v2_error - avg_v4_error:.2f}")

improved = sum(1 for c in cases_49_99 if c['improvement'] > 0)
print(f"\nCases improved: {improved}/{len(cases_49_99)} ({improved/len(cases_49_99)*100:.1f}%)")