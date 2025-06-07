# Legacy System Reverse Engineering Notes

## Confirmed Patterns
*None yet - analysis in progress*

## Suspected Patterns
- **Base per diem around $100/day** - Initial analysis shows this might be valid (Lisa's claim)
- **Mileage tiers exist** - Clear evidence of tiered pricing:
  - 0-100 miles: ~$1009 avg
  - 100-200 miles: ~$1102 avg  
  - 200-400 miles: ~$1153 avg
  - 400-600 miles: ~$1323 avg
  - 600-800 miles: ~$1411 avg
  - 800-1000 miles: ~$1573 avg
- **Low receipt penalty** - Trips with <$50 receipts average only $553.74 total

## Edge Cases & Anomalies
- Case 995: 1 day, 1082 miles, $1809.49 receipts → $446.94 (expected ~$1421 based on model)
- Case 114: 5 days, 195.73 miles, $1228.49 receipts → $511.23 (expected ~$1233)
- Several cases with >$600 error suggest different calculation modes

## Failed Hypotheses
- **5-day bonus does NOT exist** - 5-day trips average $1272.59 vs $1358.77 for others
- **Efficiency bonus (180-220 miles/day) not clearly evident** - Those trips average $1318.86 vs $1350.41 for others

## Technical Discoveries
- Random Forest achieves R²=0.936, Gradient Boosting R²=0.933
- Feature importance: receipts (64.5%), days (19.2%), miles (13.8%)
- 430 cases have >$50 error, suggesting complex non-linear patterns or bugs
- Average absolute error with best model: $58.83

## Interview Claims Status
- **Base per diem $100/day (Lisa)**: Status: LIKELY - Further analysis needed
- **5-day bonus (Lisa)**: Status: REFUTED - Data shows opposite pattern
- **Mileage tiers (Lisa)**: Status: CONFIRMED - Clear tier structure visible
- **Efficiency bonus 180-220 mpd (Kevin)**: Status: REFUTED - No clear advantage
- **Low receipt penalty (Multiple)**: Status: CONFIRMED - <$50 receipts penalized
- **Rounding bug .49/.99 (Lisa)**: Status: UNKNOWN - Needs testing
- **Magic numbers like $847 (Marcus)**: Status: UNKNOWN - Needs investigation

## Deep Statistical Analysis Findings (Phase 1.1)

### 1.1.1 Univariate Distribution Analysis
- **Reimbursement is NOT normally distributed** (Shapiro-Wilk p < 0.001)
- **Decimal patterns**: Top endings are .00 (26 cases), .99 (14 cases), .49 (8 cases)
- **Magic numbers**: No exact $847 found, but 3 cases near $847 (±10)
- **Skewness**: Reimbursement skewed right (0.81), suggesting upper tail outliers

### 1.1.2 Temporal/Sequential Pattern Analysis  
- **No significant temporal trends** detected in case sequence
- **Autocorrelation near zero** at all lags - cases appear independent
- **No evidence of system changes** over time (no changepoints detected)

### 1.1.3 Non-Linear Relationships
- **Receipts have highest mutual information** (0.89) with reimbursement
- **Strong non-linear pattern** between receipts and reimbursement
- **Polynomial (degree 2) fits better** than linear for most relationships
- **Cost per mile shows inverse relationship** with reimbursement

### 1.1.4 Segmentation & Regime Detection
- **4 distinct clusters identified** via K-means:
  - Cluster 0: Short trips, low miles, low receipts (mean reimb: $387)
  - Cluster 1: Medium trips, medium activity (mean reimb: $1053)  
  - Cluster 2: Long trips, high miles (mean reimb: $1687)
  - Cluster 3: High receipt trips (mean reimb: $1821)
- **Decision tree suggests receipts > $844** is primary split

### 1.1.5 Conditional Distribution Analysis
- **Trip length dramatically affects distribution shape**:
  - Short trips (1-3 days): tight distribution around $400-600
  - Long trips (11+ days): wide distribution $1000-2500
- **Efficiency buckets show distinct patterns**:
  - Low efficiency (<50 mpd): bimodal distribution
  - High efficiency (100-200 mpd): normal-like distribution

### 1.1.6 Statistical Hypothesis Testing Results
- **5-day bonus: REJECTED** (p=0.82, no significant difference)
- **Efficiency sweet spot: NOT CONFIRMED** (p=0.47, no advantage for 180-220 mpd)
- **Receipt threshold at $50: CONFIRMED** (p<0.001, mean diff = $797)
- **Rounding bias for .49/.99: NOT SIGNIFICANT** (p>0.05)

### 1.1.7 Outlier Characterization
- **43 combined outliers** identified (4.3% of data)
- **Outlier profile**: Very short trips (1-2 days) with high receipts
- **Top outlier**: Case 696 (1 day, 27 miles, $1987 receipts → $669 reimb)

### 1.1.8 Missing Pattern Analysis  
- **Only 28.8% coverage** of possible input combinations
- **Suspicious gaps** in reimbursement: $1653-1754, $1435-1519
- **Decimal distribution NOT uniform** (p<0.001) - suggests calculation artifacts

### Key Insights for Model Development
1. Strong non-linearity requires ensemble/NN approaches
2. Receipt amount is dominant feature - focus feature engineering here
3. 4 natural segments suggest mixture model approach
4. Low receipt penalty at $50 must be explicitly modeled
5. No temporal patterns - can use all data for training
