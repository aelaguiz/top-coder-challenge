# Reverse Engineering Legacy Reimbursement System - Phased Approach

## 📝 Knowledge Management Process

### Continuous Learning Documentation
Throughout this reverse engineering process, we will maintain a living document `notes.md` to capture all insights, patterns, and learnings as they emerge.

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
- ✅ After each analysis phase completion
- ✅ When discovering a new pattern
- ✅ When confirming/refuting an interview claim
- ✅ When encountering unexpected behavior
- ✅ After model training results
- ✅ When identifying high-error cases

**Review Schedule:**
- Before starting each new phase
- When stuck on improving accuracy
- Before final implementation
- When writing documentation

This ensures no insight is lost and patterns can be cross-referenced throughout the project.

---

## Executive Summary
Based on initial analysis of 1,000 public cases, we have sufficient data to attempt multiple regression approaches including neural networks. The system shows complex non-linear patterns with 430+ cases having >$50 error using gradient boosting, suggesting hidden rules or edge cases.

## Phase 1: Deep Data Analysis & Feature Engineering (Days 1-2)

### 1.1 Statistical Analysis - Detailed Plan

#### 1.1.1 Univariate Distribution Analysis
**Objective**: Understand the individual behavior of each variable and identify anomalies

**Reimbursement Distribution**
- Check for multimodality (multiple peaks suggesting different calculation modes)
- Identify natural breakpoints/clusters in reimbursement amounts
- Look for "magic numbers" - frequently occurring exact values
- Analyze decimal patterns (.00, .49, .99 endings per interviews)
- Outlier analysis: values beyond 3 standard deviations
- Skewness and kurtosis to understand distribution shape

**Input Variable Distributions**
- Days: Check if certain durations are over/under-represented
- Miles: Look for clustering around specific values (100, 200, 500 mile boundaries)
- Receipts: Identify suspicious gaps or concentrations
- Cross-reference with interview claims (e.g., "$847 is lucky")

**Implementation**:
```python
# Detailed distribution analysis
- Kernel Density Estimation (KDE) plots for smooth distribution visualization
- Q-Q plots to check normality assumptions
- Histogram with various bin sizes to catch patterns
- Empirical Cumulative Distribution Function (ECDF) analysis
- Statistical tests: Shapiro-Wilk, Anderson-Darling
```

#### 1.1.2 Temporal/Sequential Pattern Analysis
**Objective**: Detect if case ordering reveals system behavior changes

**Analyses**:
- Rolling statistics (mean, std) with different window sizes (10, 50, 100 cases)
- Autocorrelation analysis of reimbursements
- Check for seasonal patterns if cases have implicit time ordering
- Trend analysis: is the system becoming more/less generous over time?
- Changepoint detection to identify system updates/modifications
- Spectral analysis for periodic patterns

**Implementation**:
```python
# Time series techniques despite no explicit timestamps
- ARIMA modeling on case sequence
- STL decomposition (Seasonal-Trend-Loess)
- Fourier analysis for hidden periodicities
- CUSUM charts for detecting shifts in behavior
```

#### 1.1.3 Non-Linear Relationship Discovery
**Objective**: Uncover complex relationships beyond linear correlation

**Techniques**:
- Mutual Information scores between variables
- Distance correlation (captures non-linear dependencies)
- Maximal Information Coefficient (MIC)
- Scatter plot matrices with LOESS smoothing
- 3D visualizations for three-way interactions
- Partial dependence plots

**Specific Investigations**:
- Polynomial relationships (receipts², miles³, etc.)
- Logarithmic scaling (log(miles) vs reimbursement)
- Threshold effects (sudden changes at specific values)
- Interaction heatmaps (days×miles, miles×receipts)
- Ratio analysis (reimbursement/receipts vs other variables)

#### 1.1.4 Segmentation & Regime Detection
**Objective**: Identify if different "modes" or "regimes" exist

**Clustering Analysis**:
- K-means with elbow method for optimal clusters
- DBSCAN for density-based clustering
- Gaussian Mixture Models for probabilistic clustering
- Hierarchical clustering to understand relationships
- Self-Organizing Maps (SOM) for visualization

**Regime Detection**:
- Hidden Markov Models to detect state changes
- Decision tree splits to find natural breakpoints
- Isolation Forest for anomaly detection
- Local Outlier Factor (LOF) analysis

#### 1.1.5 Conditional Distribution Analysis
**Objective**: Understand how distributions change based on conditions

**Analyses**:
- Reimbursement distribution conditioned on:
  - Trip length buckets (1-3, 4-6, 7-10, 11+ days)
  - Efficiency levels (<50, 50-100, 100-200, 200+ miles/day)
  - Receipt categories (<$100, $100-500, $500-1000, $1000+)
- Quantile regression to understand conditional relationships
- Copula analysis for dependency structures

#### 1.1.6 Statistical Hypothesis Testing
**Objective**: Rigorously test interview claims

**Tests to Perform**:
1. **5-day bonus**: 
   - t-test comparing 5-day vs adjacent durations (4 and 6 days)
   - Control for other variables using ANCOVA
   - Propensity score matching to create comparable groups

2. **Efficiency sweet spot (180-220 miles/day)**:
   - ANOVA across efficiency buckets
   - Tukey's HSD for pairwise comparisons
   - Non-parametric alternatives (Kruskal-Wallis)

3. **Receipt thresholds**:
   - Breakpoint regression to find exact penalty points
   - Piecewise linear regression
   - Regression discontinuity design

4. **Rounding patterns (.49/.99)**:
   - Chi-square test for digit frequency
   - Benford's Law analysis
   - Exact binomial tests

#### 1.1.7 Outlier Characterization
**Objective**: Understand extreme cases as they may reveal edge rules

**Approach**:
- Mahalanobis distance for multivariate outliers
- Cook's distance from initial regression
- SHAP values to explain individual predictions
- Manual inspection of top/bottom 5% cases
- Create "outlier profiles" with common characteristics

**Documentation**:
- Maintain anomaly log with hypotheses
- Cross-reference with interview claims
- Look for systematic patterns in outliers

#### 1.1.8 Missing Pattern Analysis
**Objective**: What's NOT in the data might be informative

**Investigations**:
- Are certain combinations of inputs missing?
- Gaps in reimbursement values (never see $X)
- Impossible or avoided input combinations
- Compare actual vs expected distributions

### 1.2 Feature Engineering
**Base Features:**
- miles_per_day (efficiency metric)
- receipts_per_day (spending rate)
- total_cost_per_mile (receipts/miles ratio)

**Categorical Features:**
- trip_length_category (short: 1-3, medium: 4-7, long: 8+)
- efficiency_category (based on miles/day buckets)
- receipt_category (very_low, low, medium, medium_high, high, very_high)

**Interaction Features:**
- days × miles (total trip effort)
- days × receipts (total trip cost)
- miles × receipts_per_mile
- efficiency_score × trip_length

**Polynomial Features:**
- Consider up to degree 3 for key variables
- Focus on receipts (most important per initial analysis)

### 1.3 Interview Hypothesis Testing
Test each claim systematically:
1. **Base per diem**: $100/day (Lisa) - appears valid
2. **5-day bonus**: FALSE per initial analysis, but check subgroups
3. **Mileage tiers**: Confirmed with clear breakpoints
4. **Efficiency bonus**: 180-220 miles/day (Kevin) - needs refinement
5. **Receipt penalties**: Low amounts (<$50) confirmed
6. **"Magic combinations"**: Search for specific value patterns
7. **Rounding bugs**: Check receipts ending in .49/.99

## Phase 2: Model Development Strategy (Days 3-4)

### 2.1 Baseline Models
1. **Rule-based system** incorporating confirmed patterns:
   - Base per diem calculation
   - Mileage tier system
   - Receipt penalty/bonus rules
   
2. **Linear regression** with engineered features as sanity check

### 2.2 Traditional ML Models
1. **Random Forest** (current best: R²=0.936)
   - Tune hyperparameters extensively
   - Extract rule patterns from trees
   
2. **XGBoost/LightGBM**
   - Often better than standard gradient boosting
   - Good at capturing complex interactions

3. **Support Vector Regression**
   - With RBF kernel for non-linearity
   - May capture different patterns

### 2.3 Neural Network Approach
**Architecture Proposal:**
```
Input Layer (3-30 features depending on engineering)
Hidden Layer 1: 64 neurons, ReLU, Dropout(0.2)
Hidden Layer 2: 32 neurons, ReLU, Dropout(0.2)
Hidden Layer 3: 16 neurons, ReLU
Output Layer: 1 neuron (reimbursement amount)
```

**Training Strategy:**
- 70/15/15 train/validation/test split
- K-fold cross-validation (k=5)
- Early stopping with patience=20
- L2 regularization
- Batch normalization
- Learning rate scheduling

### 2.4 Ensemble Approach
Combine predictions from:
- Best tree-based model
- Neural network
- Rule-based system
Using weighted average or stacking

## Phase 3: Error Analysis & Edge Case Handling (Days 5-6)

### 3.1 High-Error Case Analysis
- Cluster high-error cases to find patterns
- Manual inspection of top 50 worst predictions
- Look for:
  - Boundary conditions
  - "Magic numbers" mentioned in interviews
  - Possible data entry errors or system bugs

### 3.2 Segmentation Strategy
- Build separate models for different trip types if patterns emerge
- Consider:
  - Short vs. long trips
  - High vs. low efficiency trips
  - Business travel patterns (sales vs. others)

### 3.3 Post-Processing Rules
- Implement guardrails based on business logic
- Handle edge cases identified in error analysis
- Add "bug replication" for any systematic errors found

## Phase 4: Implementation & Validation (Days 7-8)

### 4.1 Model Selection
- Compare all approaches on consistent holdout set
- Consider both accuracy and interpretability
- Ensemble if single model doesn't achieve >95% exact matches

### 4.2 Implementation Options
1. **Python-based** (recommended):
   - Use trained model with pickle/joblib
   - Fast inference, easy to maintain
   
2. **Pure calculation** (if patterns are simple enough):
   - Implement discovered rules directly
   - More transparent but less flexible

3. **Hybrid approach**:
   - Rules for common cases
   - ML model for edge cases

### 4.3 Validation Strategy
- Test on full public dataset
- Sensitivity analysis on input ranges
- Stress test edge cases
- Compare against employee anecdotes

## Phase 5: Optimization & Refinement (Day 9)

### 5.1 Fine-tuning
- Optimize for exact matches (±$0.01) not just MAE
- Custom loss function prioritizing exact matches
- Threshold-based adjustments

### 5.2 Final Testing
- Run eval.sh iteratively
- Target 100% exact matches on public cases
- Document any irreconcilable cases

## Risk Mitigation

### Technical Risks
1. **Overfitting**: Use strong regularization and cross-validation
2. **Hidden variables**: The system might use data we don't have (date, user ID, etc.)
3. **Non-deterministic behavior**: Some randomness might be intentional

### Approach Risks
1. **Over-engineering**: Start simple, add complexity only if needed
2. **Missing patterns**: Regular error analysis to catch blind spots
3. **Implementation mismatch**: Test early and often with eval.sh

## Success Metrics
- **Primary**: 100% exact matches (±$0.01) on public cases
- **Secondary**: <$1 average error on any misses
- **Stretch**: Discover and document all business rules

## Recommended Approach
Given the analysis showing R²=0.936 with tree methods and 430 high-error cases, I recommend:

1. **Start with XGBoost** with extensive feature engineering
2. **Parallel development** of neural network
3. **Deep dive** on high-error cases for rule discovery
4. **Ensemble** if needed for final accuracy push

The 1,000 samples are sufficient for this approach with proper regularization and validation strategies.