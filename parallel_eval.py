#!/usr/bin/env python3
"""
Parallel evaluation script for the reimbursement system.
Runs test cases in parallel for much faster evaluation.
"""

import json
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import sys
from collections import Counter

def run_single_case(case_data):
    """Run a single test case and return results."""
    idx, days, miles, receipts, expected = case_data
    
    try:
        # Run the prediction
        result = subprocess.run(
            ['./run.sh', str(days), str(miles), str(receipts)],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            try:
                actual = float(output)
                error = abs(actual - expected)
                return {
                    'index': idx,
                    'success': True,
                    'expected': expected,
                    'actual': actual,
                    'error': error,
                    'days': days,
                    'miles': miles,
                    'receipts': receipts
                }
            except ValueError:
                return {
                    'index': idx,
                    'success': False,
                    'error_msg': f"Invalid output: {output}"
                }
        else:
            return {
                'index': idx,
                'success': False,
                'error_msg': f"Script failed: {result.stderr}"
            }
    except subprocess.TimeoutExpired:
        return {
            'index': idx,
            'success': False,
            'error_msg': "Timeout"
        }
    except Exception as e:
        return {
            'index': idx,
            'success': False,
            'error_msg': str(e)
        }

def main():
    print("🧾 Parallel Black Box Challenge Evaluation")
    print("=" * 50)
    print()
    
    # Load test cases
    print("Loading test cases...")
    with open('public_cases.json', 'r') as f:
        cases = json.load(f)
    
    # Prepare test data
    test_data = []
    for i, case in enumerate(cases):
        inp = case['input']
        test_data.append((
            i,
            inp['trip_duration_days'],
            inp['miles_traveled'],
            inp['total_receipts_amount'],
            case['expected_output']
        ))
    
    print(f"Loaded {len(test_data)} test cases")
    print(f"Running with {multiprocessing.cpu_count()} CPUs...")
    print()
    
    # Run tests in parallel
    start_time = time.time()
    results = []
    errors = []
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Submit all tasks
        future_to_case = {executor.submit(run_single_case, case): case for case in test_data}
        
        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_case)):
            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{len(test_data)} cases processed...", file=sys.stderr)
            
            result = future.result()
            if result['success']:
                results.append(result)
            else:
                errors.append(result)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds!")
    
    # Calculate metrics
    if not results:
        print("\n❌ No successful test cases!")
        return
    
    # Sort results by index to maintain order
    results.sort(key=lambda x: x['index'])
    
    # Calculate statistics
    exact_matches = sum(1 for r in results if r['error'] < 0.01)
    close_matches = sum(1 for r in results if r['error'] < 1.0)
    total_error = sum(r['error'] for r in results)
    avg_error = total_error / len(results)
    max_error_result = max(results, key=lambda x: x['error'])
    
    # Calculate percentages
    exact_pct = exact_matches * 100 / len(results)
    close_pct = close_matches * 100 / len(results)
    
    print("\n✅ Evaluation Complete!")
    print()
    print("📈 Results Summary:")
    print(f"  Total test cases: {len(cases)}")
    print(f"  Successful runs: {len(results)}")
    print(f"  Exact matches (±$0.01): {exact_matches} ({exact_pct:.1f}%)")
    print(f"  Close matches (±$1.00): {close_matches} ({close_pct:.1f}%)")
    print(f"  Average error: ${avg_error:.2f}")
    print(f"  Maximum error: ${max_error_result['error']:.2f}")
    print()
    
    # Calculate score (same as eval.sh)
    score = avg_error * 100 + (len(cases) - exact_matches) * 0.1
    print(f"🎯 Your Score: {score:.2f} (lower is better)")
    print()
    
    # Show top errors
    print("💡 Top 10 high-error cases:")
    top_errors = sorted(results, key=lambda x: x['error'], reverse=True)[:10]
    for r in top_errors:
        print(f"  Case {r['index'] + 1}: {r['days']} days, {r['miles']:.2f} miles, ${r['receipts']:.2f} receipts")
        print(f"    Expected: ${r['expected']:.2f}, Got: ${r['actual']:.2f}, Error: ${r['error']:.2f}")
    
    # Show errors if any
    if errors:
        print(f"\n⚠️  {len(errors)} cases failed:")
        for e in errors[:5]:
            print(f"  Case {e['index'] + 1}: {e['error_msg']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
    
    # Save detailed results
    with open('parallel_eval_results.json', 'w') as f:
        json.dump({
            'summary': {
                'total_cases': len(cases),
                'successful_runs': len(results),
                'exact_matches': exact_matches,
                'close_matches': close_matches,
                'average_error': avg_error,
                'max_error': max_error_result['error'],
                'score': score,
                'elapsed_seconds': elapsed
            },
            'results': results,
            'errors': errors
        }, f, indent=2)
    
    print("\n📄 Detailed results saved to parallel_eval_results.json")

if __name__ == '__main__':
    main()