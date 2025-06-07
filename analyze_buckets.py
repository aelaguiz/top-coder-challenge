#!/usr/bin/env python3
"""Analyze error patterns by bucketing similar cases."""

import json
import numpy as np
from collections import defaultdict

# Load results
with open('parallel_eval_results.json', 'r') as f:
    data = json.load(f)

results = data['results']

# Define buckets
buckets = defaultdict(list)

for r in results:
    days = r['days']
    miles = r['miles']
    receipts = r['receipts']
    error = r['error']
    expected = r['expected']
    actual = r['actual']
    
    # Calculate derived features
    miles_per_day = miles / days if days > 0 else 0
    
    # Bucket by receipt ranges and other characteristics
    if receipts < 30:
        if days == 1:
            bucket_name = "Low Receipt - Single Day"
        elif days <= 3:
            bucket_name = "Low Receipt - Short Trip (2-3 days)"
        elif days <= 6:
            bucket_name = "Low Receipt - Medium Trip (4-6 days)"
        else:
            bucket_name = "Low Receipt - Long Trip (7+ days)"
    elif receipts < 100:
        bucket_name = "Medium-Low Receipt ($30-100)"
    elif receipts < 500:
        bucket_name = "Medium Receipt ($100-500)"
    elif receipts < 1000:
        bucket_name = "Medium-High Receipt ($500-1000)"
    elif receipts < 1500:
        bucket_name = "High Receipt ($1000-1500)"
    else:
        bucket_name = "Very High Receipt ($1500+)"
    
    # Add special buckets for high efficiency
    if miles_per_day >= 180 and miles_per_day <= 220:
        bucket_name += " [Efficiency Sweet Spot]"
    
    buckets[bucket_name].append({
        'error': error,
        'pct_error': (error / expected * 100) if expected > 0 else 0,
        'expected': expected,
        'actual': actual,
        'days': days,
        'miles': miles,
        'receipts': receipts,
        'miles_per_day': miles_per_day,
        'over_predicted': actual > expected
    })

# Analyze each bucket
print("# Error Analysis by Bucket\n")
print(f"Total cases: {len(results)}")
print(f"Average error: ${data['summary']['average_error']:.2f}")
print(f"Exact matches: {data['summary']['exact_matches']} ({data['summary']['exact_matches']/10:.1f}%)\n")

# Sort buckets by average error
bucket_stats = []
for bucket_name, cases in buckets.items():
    if cases:
        errors = [c['error'] for c in cases]
        avg_error = np.mean(errors)
        max_error = max(errors)
        exact_matches = sum(1 for e in errors if e < 0.01)
        over_predicted = sum(1 for c in cases if c['over_predicted'])
        
        bucket_stats.append({
            'name': bucket_name,
            'count': len(cases),
            'avg_error': avg_error,
            'max_error': max_error,
            'exact_matches': exact_matches,
            'over_predicted_pct': over_predicted / len(cases) * 100,
            'cases': cases
        })

bucket_stats.sort(key=lambda x: x['avg_error'], reverse=True)

# Print bucket analysis
for bs in bucket_stats:
    print(f"\n## {bs['name']}")
    print(f"- Cases: {bs['count']}")
    print(f"- Avg Error: ${bs['avg_error']:.2f}")
    print(f"- Max Error: ${bs['max_error']:.2f}")
    print(f"- Exact Matches: {bs['exact_matches']} ({bs['exact_matches']/bs['count']*100:.1f}%)")
    print(f"- Over-predicted: {bs['over_predicted_pct']:.1f}%")
    
    # Show top errors in this bucket
    top_errors = sorted(bs['cases'], key=lambda x: x['error'], reverse=True)[:3]
    if top_errors[0]['error'] > 10:  # Only show if significant errors
        print(f"\nWorst cases:")
        for case in top_errors:
            print(f"  - {case['days']}d, {case['miles']:.0f}mi, ${case['receipts']:.2f} receipts")
            print(f"    Expected: ${case['expected']:.2f}, Got: ${case['actual']:.2f}, Error: ${case['error']:.2f}")

# Find patterns
print("\n\n# Key Patterns\n")

# Check if our low receipt formula is working
low_receipt_buckets = [b for b in bucket_stats if "Low Receipt" in b['name']]
if low_receipt_buckets:
    print("## Low Receipt Formula Performance (<$30)")
    for b in low_receipt_buckets:
        print(f"- {b['name']}: Avg error ${b['avg_error']:.2f}")
    print("\nConclusion: Our $100/day + $0.75/mile formula needs adjustment based on trip length!")

# Look for systematic over/under prediction
print("\n## Systematic Prediction Bias")
for b in bucket_stats:
    if b['over_predicted_pct'] > 80:
        print(f"- {b['name']}: Over-predicting {b['over_predicted_pct']:.0f}% of the time")
    elif b['over_predicted_pct'] < 20:
        print(f"- {b['name']}: Under-predicting {100-b['over_predicted_pct']:.0f}% of the time")