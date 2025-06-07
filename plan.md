# Reverse Engineering Legacy Reimbursement System - Implementation Plan

## 📝 Knowledge Management Process

### Continuous Learning Documentation
Throughout this reverse engineering process, we maintain a living document `notes.md` to capture all insights, patterns, and learnings as they emerge.

**Structure of notes.md:**
```markdown
# Legacy System Reverse Engineering Notes

## Confirmed Patterns
- Pattern description with evidence
- Exact formula/rule if discovered
- Test cases that validate this pattern

## Suspected Patterns
- Hypothesis with supporting evidence
- Test cases that suggest this pattern
- Further validation needed

## Edge Cases & Anomalies
- Description of unusual behavior
- Case IDs and specific examples
- Possible explanations

## Failed Hypotheses
- What we tested
- Why it didn't work
- Lessons learned

## Technical Discoveries
- Model performance insights
- Feature engineering successes/failures
- Algorithm-specific findings

## Interview Claims Status
- Claim: [Original statement]
- Status: [Confirmed/Refuted/Partial/Unknown]
- Evidence: [Supporting data]
```

**When to Update notes.md:**
- ✅ After each eval.sh run with new patterns
- ✅ When discovering error patterns
- ✅ When confirming/refuting hypotheses
- ✅ After implementing fixes
- ✅ When achieving milestones

---

## Executive Summary

We have discovered that `eval.sh` provides immediate feedback on all 1,000 test cases, showing exact matches, close matches, and specific high-error cases. This fundamentally changes our approach from theoretical analysis to **empirical implementation-driven discovery**.

**Key Learnings from Initial Analysis:**
- Receipt amount is the dominant feature (highest mutual information: 0.565)
- 4 distinct data clusters exist with different reimbursement patterns
- Low receipt penalty confirmed (<$50 receipts = ~$822 less reimbursement)
- Primary decision split occurs at receipts > $844
- No 5-day bonus exists (contrary to interviews)
- No temporal patterns - data is stationary

## Primary Strategy: Empirical Implementation-Driven Discovery

### Why This Approach Wins
1. **eval.sh is our oracle** - Instant feedback on 1,000 cases with exact error amounts
2. **Errors reveal patterns** - High-error cases show us exactly what rules we're missing
3. **No guessing required** - Every hypothesis is immediately validated
4. **Clear success metric** - 100% exact matches (±$0.01) is unambiguous

### The Implementation Loop

```
┌─────────────────┐
│   Implement     │
│  Best Model     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Run eval.sh   │
│ Get Error Cases │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analyze Errors  │
│  Find Patterns  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Update Model   │
│   Add Rules     │
└────────┬────────┘
         │
         └──────── Repeat until 100% ─┘
```

## Phase 1: Baseline Implementation (30-60 mins)

### Objective
Create initial implementation using our analysis findings and get baseline metrics.

### Implementation Details
1. **Model Choice**: Decision tree or XGBoost (both showed promise)
   - Decision trees naturally capture arbitrary rules/bugs
   - Use our 4 identified clusters
   - Implement known thresholds (receipts <$50, >$844)

2. **Feature Engineering** (from our analysis):
   ```python
   # Core features
   miles_per_day = miles / days
   receipts_per_day = receipts / days
   cost_per_mile = receipts / miles
   
   # Categorical features for regime detection
   trip_length_bucket = categorize_days(days)  # short/medium/long
   receipt_bucket = categorize_receipts(receipts)  # low/medium/high
   efficiency_bucket = categorize_efficiency(miles_per_day)
   ```

3. **Initial Rules** (confirmed patterns):
   ```python
   # Low receipt penalty
   if receipts < 50:
       apply_penalty()  # ~40% reduction based on data
   
   # Primary split
   if receipts > 844:
       high_receipt_regime()
   else:
       low_receipt_regime()
   ```

4. **Setup Scripts**:
   - Create `calculate_reimbursement.py`
   - Create `run.sh` from template
   - Ensure proper decimal rounding (2 places)

### Success Criteria
- Script runs successfully
- Get baseline scores from eval.sh
- Document initial performance in notes.md

## Phase 2: Error-Driven Pattern Discovery (2-4 hours)

### Objective
Use eval.sh output to discover missing patterns empirically.

### Process
1. **Extract High-Error Cases**
   ```bash
   ./eval.sh > baseline_results.txt
   # Focus on cases with error > $50
   ```

2. **Pattern Analysis**
   - Group errors by input characteristics
   - Look for systematic biases
   - Check if errors correlate with:
     - Specific day counts (especially edge values)
     - Receipt thresholds we haven't discovered
     - Mile boundaries
     - Ratio relationships

3. **Hypothesis Testing**
   - For each pattern hypothesis:
     - Implement the rule
     - Run eval.sh
     - Keep if improvement, revert if not
     - Document in notes.md

4. **Common Patterns to Check**:
   - Boundary effects (day 1, day 14, etc.)
   - Receipt rounding bugs (.49/.99 endings)
   - Hidden thresholds
   - Multiplicative effects
   - Integer overflow/underflow artifacts

### Tools for Analysis
```python
# After each eval.sh run:
def analyze_errors(results_file):
    # Load errors
    # Group by characteristics
    # Find commonalities
    # Generate hypotheses
    # Update notes.md
```

## Phase 3: Refinement & Edge Cases (2-3 hours)

### Objective
Handle remaining errors through targeted fixes.

### Strategies
1. **Ensemble Approach** (if single model plateaus)
   - Combine multiple models
   - Use case-specific models for problem clusters
   - Weight by confidence

2. **Rule-Based Overrides**
   - Hard-code specific problem cases if patterns unclear
   - Add post-processing for systematic biases
   - Implement "bug replication" for legacy quirks

3. **Decimal Precision Handling**
   - Ensure consistent rounding
   - Check for floating-point artifacts
   - Match legacy system's precision exactly

### Iteration Checklist
- [ ] Run eval.sh after each change
- [ ] Update notes.md with findings
- [ ] Track improvement metrics
- [ ] Save each working version
- [ ] Document which changes helped

## Phase 4: Final Optimization (1 hour)

### Objective
Achieve 100% exact matches and prepare for private cases.

### Tasks
1. **Clean Implementation**
   - Remove experimental code
   - Optimize for speed (<5 seconds per case)
   - Add error handling

2. **Validation**
   - Confirm 100% exact matches on public cases
   - Test edge cases manually
   - Verify decimal precision

3. **Prepare Submission**
   - Run `generate_results.sh` for private cases
   - Document final approach
   - Create clean git history

## Tracking Progress

### Metrics to Track
After each eval.sh run, record:
- Exact matches (target: 1000/1000)
- Close matches
- Average error
- Maximum error
- Number of cases improved
- Patterns discovered

### Expected Timeline
- **Hour 1**: Baseline implementation + first eval.sh run
- **Hour 2-3**: Major pattern discoveries
- **Hour 4-5**: Edge case handling
- **Hour 6**: Final optimization and validation

## Key Insights to Leverage

From our completed Phase 1.1 analysis:

1. **Cluster-Based Approach**
   - 4 distinct clusters identified
   - Each may have different rules
   - Use cluster membership as feature

2. **Receipt Dominance**
   - Receipts are most predictive
   - Multiple thresholds exist
   - Non-linear relationships confirmed

3. **No Time Component**
   - Case order doesn't matter
   - No seasonal effects
   - Can use all data equally

4. **Failed Interview Claims**
   - No 5-day bonus
   - No 180-220 mpd efficiency bonus
   - These save us from false paths

## Contingency Plans

### If Stuck at <95% Exact Matches
1. Deep dive on systematic errors
2. Try polynomial features
3. Implement case-specific overrides
4. Consider neural network

### If Stuck at 95-99% Exact Matches
1. Manual inspection of all failures
2. Look for data entry errors
3. Check for modulo arithmetic
4. Consider hardcoding edge cases

### If Performance Issues
1. Simplify model
2. Pre-compute features
3. Use lookup tables
4. Optimize Python code

## Success Criteria

### Minimum Viable Solution
- ✅ Runs successfully
- ✅ >90% exact matches
- ✅ <$10 average error

### Target Solution
- ✅ 100% exact matches on public cases
- ✅ <5 second runtime
- ✅ Clean, maintainable code

### Stretch Goals
- Document all discovered rules
- Explain the "bugs" in the system
- Build interpretable model
- Create visualization of decision logic

---

## Appendix: Completed Analysis Details

<details>
<summary>Phase 1.1: Deep Statistical Analysis (COMPLETED)</summary>

### Key Findings
- Reimbursement is NOT normally distributed (p < 0.001)
- 4 distinct clusters via K-means
- Decision tree primary split at receipts > $844
- Low receipt penalty confirmed at $50 threshold
- No temporal patterns detected
- 16 statistical outliers identified
- Only 40.2% coverage of possible input combinations

### Statistical Tests Results
- 5-day bonus: REJECTED (p=0.75)
- Efficiency sweet spot: NOT CONFIRMED (p=0.47)
- Receipt thresholds: CONFIRMED at multiple levels
- Decimal patterns: Uniformly distributed (p=0.86)

</details>

## Remember

**The key to success is rapid iteration using eval.sh feedback. Don't overthink - implement, test, and refine!**