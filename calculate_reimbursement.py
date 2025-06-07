#!/usr/bin/env python3
"""
Baseline implementation for travel reimbursement calculation.
Based on Phase 1.1 analysis findings.
"""

import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib
import os
import logging
from datetime import datetime

# Set up logging
log_file = 'reimbursement_calculation.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file)
        # Removed stderr handler to keep eval.sh output clean
    ]
)

# Check if we need to train the model
MODEL_FILE = 'reimbursement_model.pkl'
FEATURES_FILE = 'feature_columns.json'

# Load expected values for logging/debugging
with open('public_cases.json', 'r') as f:
    PUBLIC_CASES = json.load(f)
    EXPECTED_OUTPUTS = {
        (case['input']['trip_duration_days'], 
         case['input']['miles_traveled'], 
         case['input']['total_receipts_amount']): case['expected_output']
        for case in PUBLIC_CASES
    }

def create_features(days, miles, receipts):
    """Create engineered features based on our analysis."""
    # Avoid division by zero
    days = max(days, 0.1)
    miles = max(miles, 0.1)
    receipts = max(receipts, 0.01)
    
    # Core features from analysis
    features = {
        'days': days,
        'miles': miles,
        'receipts': receipts,
        'miles_per_day': miles / days,
        'receipts_per_day': receipts / days,
        'cost_per_mile': receipts / miles,
        'days_squared': days ** 2,
        'miles_squared': miles ** 2,
        'receipts_squared': receipts ** 2,
        'days_miles': days * miles,
        'days_receipts': days * receipts,
        'miles_receipts': miles * receipts,
        
        # Categorical indicators based on our cluster analysis
        'is_short_trip': 1 if days <= 3 else 0,
        'is_medium_trip': 1 if 4 <= days <= 6 else 0,
        'is_long_trip': 1 if days >= 7 else 0,
        'is_very_long_trip': 1 if days >= 11 else 0,
        
        # Receipt thresholds from analysis
        'low_receipts': 1 if receipts < 50 else 0,
        'high_receipts': 1 if receipts > 844 else 0,
        'very_high_receipts': 1 if receipts > 1500 else 0,
        
        # Efficiency categories
        'low_efficiency': 1 if miles / days < 50 else 0,
        'high_efficiency': 1 if 100 <= miles / days <= 200 else 0,
        'very_high_efficiency': 1 if miles / days > 200 else 0,
        
        # Log transforms for non-linear relationships
        'log_receipts': np.log1p(receipts),
        'log_miles': np.log1p(miles),
        'log_days': np.log1p(days),
    }
    
    return features

def train_model():
    """Train the model on public cases."""
    logging.info("Starting model training...")
    
    # Load training data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} training cases")
    
    # Create feature matrix
    features_list = []
    targets = []
    
    for i, case in enumerate(data):
        if i % 100 == 0:
            logging.info(f"Processing training case {i}/{len(data)}")
        
        inp = case['input']
        features = create_features(
            inp['trip_duration_days'],
            inp['miles_traveled'],
            inp['total_receipts_amount']
        )
        features_list.append(features)
        targets.append(case['expected_output'])
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    feature_cols = df.columns.tolist()
    X = df.values
    y = np.array(targets)
    
    logging.info(f"Created feature matrix with shape {X.shape}")
    
    # Train model - using Gradient Boosting based on our analysis
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
        loss='huber',  # Robust to outliers
        alpha=0.9
    )
    
    logging.info("Training Gradient Boosting model...")
    model.fit(X, y)
    logging.info("Model training complete")
    
    # Calculate training score
    train_score = model.score(X, y)
    logging.info(f"Training R² score: {train_score:.4f}")
    
    # Save model and feature columns
    joblib.dump(model, MODEL_FILE)
    with open(FEATURES_FILE, 'w') as f:
        json.dump(feature_cols, f)
    logging.info(f"Model saved to {MODEL_FILE}")
    
    return model, feature_cols

def predict_reimbursement(days, miles, receipts):
    """Predict reimbursement amount."""
    # Load or train model
    if os.path.exists(MODEL_FILE) and os.path.exists(FEATURES_FILE):
        logging.debug(f"Loading existing model from {MODEL_FILE}")
        model = joblib.load(MODEL_FILE)
        with open(FEATURES_FILE, 'r') as f:
            feature_cols = json.load(f)
    else:
        logging.info("No existing model found, training new model...")
        model, feature_cols = train_model()
    
    # Create features
    features = create_features(days, miles, receipts)
    
    # Ensure features are in correct order
    feature_vector = [features[col] for col in feature_cols]
    
    # Make prediction
    prediction = model.predict([feature_vector])[0]
    
    logging.debug(f"Raw prediction: {prediction:.2f} for inputs: days={days}, miles={miles}, receipts={receipts}")
    
    # Apply known business rules from analysis
    # Low receipt penalty
    if receipts < 50:
        # Our analysis showed severe penalty for low receipts
        old_prediction = prediction
        prediction *= 0.6  # Approximate adjustment based on data
        logging.debug(f"Applied low receipt penalty: {old_prediction:.2f} -> {prediction:.2f}")
    
    # Ensure non-negative
    prediction = max(0, prediction)
    
    # Round to 2 decimal places
    final_result = round(prediction, 2)
    logging.debug(f"Final result: {final_result}")
    return final_result

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python calculate_reimbursement.py <days> <miles> <receipts>")
        sys.exit(1)
    
    try:
        days = int(sys.argv[1])
        miles = int(sys.argv[2])
        receipts = float(sys.argv[3])
        
        # Look up expected value if available
        expected = EXPECTED_OUTPUTS.get((days, miles, receipts), None)
        
        logging.info(f"Prediction request: days={days}, miles={miles}, receipts={receipts}")
        result = predict_reimbursement(days, miles, receipts)
        print(f"{result:.2f}")
        
        if expected is not None:
            error = abs(result - expected)
            error_marker = "❌" if error > 50 else "⚠️" if error > 10 else "✓"
            logging.info(f"{error_marker} Prediction: {result:.2f} | Expected: {expected:.2f} | Error: ${error:.2f}")
            
            # Log high-error cases with more detail
            if error > 50:
                logging.warning(f"HIGH ERROR CASE: days={days}, miles={miles}, receipts={receipts:.2f}")
                logging.warning(f"  Predicted: ${result:.2f}, Expected: ${expected:.2f}, Error: ${error:.2f}")
        else:
            logging.info(f"Prediction complete: {result:.2f} (no expected value available)")
        
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)