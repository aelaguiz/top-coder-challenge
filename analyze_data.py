#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

# Load the data
with open('public_cases.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame([
    {
        'days': d['input']['trip_duration_days'],
        'miles': d['input']['miles_traveled'],
        'receipts': d['input']['total_receipts_amount'],
        'reimbursement': d['expected_output']
    }
    for d in data
])

print("=== Basic Statistics ===")
print(df.describe())
print("\n=== Correlations ===")
print(df.corr())

# Create derived features based on interview insights
df['miles_per_day'] = df['miles'] / df['days']
df['receipts_per_day'] = df['receipts'] / df['days']
df['efficiency_score'] = df['miles_per_day']  # Kevin mentioned 180-220 optimal
df['is_5_day'] = (df['days'] == 5).astype(int)  # Lisa mentioned 5-day bonus
df['is_efficient'] = ((df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220)).astype(int)
df['receipt_category'] = pd.cut(df['receipts'], bins=[0, 50, 100, 300, 600, 800, 2000], labels=['very_low', 'low', 'medium', 'medium_high', 'high', 'very_high'])

# Per diem analysis
df['base_per_diem'] = df['days'] * 100  # Lisa mentioned $100/day base
df['reimbursement_minus_base'] = df['reimbursement'] - df['base_per_diem']

print("\n=== Derived Features ===")
print(df[['miles_per_day', 'receipts_per_day', 'is_5_day', 'is_efficient']].describe())

# Visualizations
plt.figure(figsize=(15, 10))

# 1. Reimbursement by trip duration
plt.subplot(2, 3, 1)
df.boxplot(column='reimbursement', by='days')
plt.title('Reimbursement by Trip Duration')
plt.suptitle('')

# 2. Miles vs Reimbursement
plt.subplot(2, 3, 2)
plt.scatter(df['miles'], df['reimbursement'], alpha=0.5)
plt.xlabel('Miles Traveled')
plt.ylabel('Reimbursement')
plt.title('Miles vs Reimbursement')

# 3. Receipts vs Reimbursement
plt.subplot(2, 3, 3)
plt.scatter(df['receipts'], df['reimbursement'], alpha=0.5)
plt.xlabel('Total Receipts')
plt.ylabel('Reimbursement')
plt.title('Receipts vs Reimbursement')

# 4. Miles per day vs Reimbursement
plt.subplot(2, 3, 4)
plt.scatter(df['miles_per_day'], df['reimbursement'], alpha=0.5)
plt.xlabel('Miles per Day')
plt.ylabel('Reimbursement')
plt.title('Efficiency vs Reimbursement')
plt.axvline(x=180, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=220, color='r', linestyle='--', alpha=0.5)

# 5. 5-day trip analysis
plt.subplot(2, 3, 5)
for days in df['days'].unique():
    subset = df[df['days'] == days]
    plt.scatter(subset.index, subset['reimbursement'], label=f'{days} days', alpha=0.6)
plt.xlabel('Case Index')
plt.ylabel('Reimbursement')
plt.title('Reimbursement by Trip Length')
plt.legend()

# 6. Receipt categories
plt.subplot(2, 3, 6)
df.boxplot(column='reimbursement', by='receipt_category')
plt.xticks(rotation=45)
plt.title('Reimbursement by Receipt Category')
plt.suptitle('')

plt.tight_layout()
plt.savefig('exploratory_analysis.png')
plt.close()

# Analyze specific patterns mentioned in interviews
print("\n=== Interview Hypothesis Testing ===")

# 1. 5-day bonus (Lisa)
five_day_avg = df[df['days'] == 5]['reimbursement'].mean()
other_avg = df[df['days'] != 5]['reimbursement'].mean()
print(f"5-day average: ${five_day_avg:.2f}")
print(f"Other days average: ${other_avg:.2f}")
print(f"5-day bonus evidence: {five_day_avg > other_avg}")

# 2. Mileage tiers (Lisa)
miles_bins = [0, 100, 200, 400, 600, 800, 1000]
df['miles_tier'] = pd.cut(df['miles'], bins=miles_bins)
print("\nMileage tier analysis:")
print(df.groupby('miles_tier')['reimbursement'].agg(['mean', 'count']))

# 3. Efficiency bonus (Kevin: 180-220 miles/day optimal)
efficient_trips = df[(df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220)]
other_trips = df[(df['miles_per_day'] < 180) | (df['miles_per_day'] > 220)]
print(f"\nEfficient trips (180-220 mpd) avg: ${efficient_trips['reimbursement'].mean():.2f}")
print(f"Other trips avg: ${other_trips['reimbursement'].mean():.2f}")

# 4. Receipt penalties for low amounts (multiple interviews)
low_receipt_trips = df[df['receipts'] < 50]
print(f"\nLow receipt trips (<$50) analysis:")
print(f"Average reimbursement: ${low_receipt_trips['reimbursement'].mean():.2f}")
print(f"Average per diem equivalent: ${(low_receipt_trips['reimbursement'] / low_receipt_trips['days']).mean():.2f}")

# Model Development
print("\n=== Model Development ===")

# Prepare features
feature_cols = ['days', 'miles', 'receipts', 'miles_per_day', 'receipts_per_day', 
                'is_5_day', 'is_efficient']
X = df[feature_cols]
y = df['reimbursement']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try different models
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'mae': mae, 'r2': r2, 'model': model}
    print(f"{name}: MAE=${mae:.2f}, R²={r2:.3f}")

# Feature importance from best model
best_model = results['GradientBoosting']['model']
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n=== Feature Importance (Gradient Boosting) ===")
print(feature_importance)

# Analyze residuals
y_pred_all = best_model.predict(X)
df['predicted'] = y_pred_all
df['residual'] = df['reimbursement'] - df['predicted']
df['abs_error'] = abs(df['residual'])

print("\n=== Residual Analysis ===")
print(f"Average absolute error: ${df['abs_error'].mean():.2f}")
print(f"Cases with error > $10: {len(df[df['abs_error'] > 10])}")
print(f"Cases with error > $50: {len(df[df['abs_error'] > 50])}")

# Save detailed analysis
df.to_csv('detailed_analysis.csv', index=False)
print("\nDetailed analysis saved to detailed_analysis.csv")

# Look for patterns in high-error cases
high_error_cases = df[df['abs_error'] > 50].sort_values('abs_error', ascending=False)
print("\n=== High Error Cases ===")
print(high_error_cases[['days', 'miles', 'receipts', 'reimbursement', 'predicted', 'abs_error']].head(10))