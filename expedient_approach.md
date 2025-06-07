# Expedient Path Forward - Data-Driven Approach

## Why This Approach?
- Uses only empirically discovered patterns (no guessing)
- Leverages confirmed thresholds and clusters
- Decision trees are interpretable and match the "buggy legacy system" profile
- Can capture non-linear relationships without assuming formulas

## Immediate Next Steps

### Step 1: Rule-Based Foundation (30 min)
Build deterministic rules for confirmed patterns:
```python
if receipts < 50:
    # Apply severe penalty (average $553 vs $1376)
    base_reimb = calculate_base(days, miles)
    return base_reimb * 0.4  # Empirically derived penalty
elif receipts > 844:
    # High receipt regime - use cluster 0/3 logic
    ...
```

### Step 2: Decision Tree Regime Detection (1 hour)
1. Train a deep decision tree (max_depth=10-15) to capture all splits
2. Extract the rules programmatically
3. Convert to if/else logic for run.sh implementation

### Step 3: Cluster-Specific Models (2 hours)
For each of the 4 identified clusters:
1. Train a simple polynomial regression
2. Fine-tune with gradient boosting for that cluster only
3. Use weighted ensemble based on cluster probability

### Step 4: Ensemble Validation (1 hour)
1. Combine rule-based + tree-based + cluster models
2. Use 80/20 train/test split
3. Target <$50 average error (currently at $58.83)

## Why This Works
- **No guessing**: Every rule comes from data analysis
- **Interpretable**: Can trace each decision back to analysis
- **Captures bugs**: Decision trees naturally capture arbitrary thresholds
- **Fast to implement**: ~5 hours to working solution

## What We're NOT Doing
- ❌ Trying to reverse-engineer mathematical formulas
- ❌ Making assumptions about business logic
- ❌ Building complex neural networks before simpler approaches
- ❌ Guessing at magic numbers or special dates

## Success Metrics
- Match known patterns (low receipt penalty, clusters)
- Achieve <5% error rate on validation set
- Rules are traceable to Phase 1.1 findings