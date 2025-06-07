#!/bin/bash

# Run eval.sh and extract key metrics
echo "Running evaluation..."
./eval.sh > eval_v2_results.txt 2>&1

# Extract key metrics
echo ""
echo "=== EVALUATION SUMMARY ==="
grep "Exact matches" eval_v2_results.txt
grep "Close matches" eval_v2_results.txt
grep "Average error" eval_v2_results.txt
grep "Maximum error" eval_v2_results.txt
grep "Your Score" eval_v2_results.txt

# Show top errors
echo ""
echo "=== TOP HIGH-ERROR CASES ==="
grep -A5 "Check these high-error cases:" eval_v2_results.txt | tail -n 10