#!/usr/bin/env python3
"""
Version 2: Implementing discoveries from Phase 1 baseline.
Key changes:
- Fixed miles to accept float values
- Testing simple formula for low receipt cases
- Removed incorrect low receipt penalty
"""

import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os
import logging

# Set up logging
log_file = 'reimbursement_calculation_v2.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file)
    ]
)

# Load expected values for logging/debugging
with open('public_cases.json', 'r') as f:
    PUBLIC_CASES = json.load(f)
    EXPECTED_OUTPUTS = {
        (case['input']['trip_duration_days'], 
         float(case['input']['miles_traveled']),  # Convert to float
         case['input']['total_receipts_amount']): case['expected_output']
        for case in PUBLIC_CASES
    }

MODEL_FILE = 'reimbursement_model_v2.pkl'
FEATURES_FILE = 'feature_columns_v2.json'

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
        
        # Categorical indicators
        'is_short_trip': 1 if days <= 3 else 0,
        'is_medium_trip': 1 if 4 <= days <= 6 else 0,
        'is_long_trip': 1 if days >= 7 else 0,
        'is_very_long_trip': 1 if days >= 11 else 0,
        
        # Updated receipt thresholds - low receipts are special!
        'very_low_receipts': 1 if receipts < 30 else 0,
        'low_receipts': 1 if receipts < 100 else 0,
        'medium_receipts': 1 if 100 <= receipts < 500 else 0,
        'high_receipts': 1 if receipts >= 500 else 0,
        'very_high_receipts': 1 if receipts >= 1000 else 0,
        
        # Efficiency categories
        'low_efficiency': 1 if miles / days < 50 else 0,
        'medium_efficiency': 1 if 50 <= miles / days < 150 else 0,
        'high_efficiency': 1 if miles / days >= 150 else 0,
        
        # Log transforms
        'log_receipts': np.log1p(receipts),
        'log_miles': np.log1p(miles),
        'log_days': np.log1p(days),
    }
    
    return features

def test_simple_formulas(days, miles, receipts):
    """Test various simple formulas that might work for low receipt cases."""
    formulas = {
        'f1_100_per_day': 100 * days,
        'f2_125_per_day': 125 * days,
        'f3_100_day_0.5_mile': 100 * days + 0.5 * miles,
        'f4_125_day_0.4_mile': 125 * days + 0.4 * miles,
        'f5_100_day_0.75_mile': 100 * days + 0.75 * miles,
        'f6_150_per_day': 150 * days,
        'f7_miles_only': miles * 1.5,
        'f8_complex': 80 * days + 0.8 * miles + 0.1 * receipts,
    }
    return formulas

def train_model():
    """Train the model on public cases."""
    logging.info("Starting model training (V2)...")
    
    # Load training data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} training cases")
    
    # Separate low and high receipt cases for different treatment
    low_receipt_data = []
    high_receipt_data = []
    
    for case in data:
        inp = case['input']
        receipts = inp['total_receipts_amount']
        if receipts < 30:
            low_receipt_data.append(case)
        else:
            high_receipt_data.append(case)
    
    logging.info(f"Low receipt cases (<$30): {len(low_receipt_data)}")
    logging.info(f"High receipt cases (>=$30): {len(high_receipt_data)}")
    
    # Analyze low receipt cases to find pattern
    if low_receipt_data:
        logging.info("\nAnalyzing low receipt cases for formula pattern...")
        best_formula = None
        best_error = float('inf')
        
        for case in low_receipt_data[:10]:  # Sample analysis
            inp = case['input']
            days = inp['trip_duration_days']
            miles = float(inp['miles_traveled'])
            receipts = inp['total_receipts_amount']
            expected = case['expected_output']
            
            formulas = test_simple_formulas(days, miles, receipts)
            
            for name, result in formulas.items():
                error = abs(result - expected)
                if error < best_error:
                    best_error = error
                    best_formula = name
            
            logging.info(f"  d={days}, m={miles:.1f}, r=${receipts:.2f} => ${expected:.2f}")
            logging.info(f"    Best formula: {best_formula} (error: ${best_error:.2f})")
    
    # Train regular model on all data for now
    features_list = []
    targets = []
    
    for i, case in enumerate(data):
        if i % 100 == 0:
            logging.info(f"Processing training case {i}/{len(data)}")
        
        inp = case['input']
        features = create_features(
            inp['trip_duration_days'],
            float(inp['miles_traveled']),  # Fixed: use float
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
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=500,  # Increased
        learning_rate=0.03,  # Decreased for better convergence
        max_depth=8,  # Increased
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42,
        loss='huber',
        alpha=0.9
    )
    
    logging.info("Training Gradient Boosting model (V2)...")
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
    """Predict reimbursement amount with special handling for low receipts."""
    
    # Special handling for very low receipt cases
    if receipts < 30:
        # Test different formulas based on our analysis
        prediction = 100 * days + 0.75 * miles  # Starting with a simple formula
        logging.info(f"Using special formula for low receipts: $100*{days} + $0.75*{miles} = ${prediction:.2f}")
    else:
        # Use ML model for other cases
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
        
        logging.debug(f"ML model prediction: {prediction:.2f} for inputs: days={days}, miles={miles}, receipts={receipts}")
    
    # Ensure non-negative
    prediction = max(0, prediction)
    
    # Round to 2 decimal places
    return round(prediction, 2)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python calculate_reimbursement_v2.py <days> <miles> <receipts>")
        sys.exit(1)
    
    try:
        days = int(sys.argv[1])
        miles = float(sys.argv[2])  # Fixed: use float instead of int
        receipts = float(sys.argv[3])
        
        # Look up expected value if available
        expected = EXPECTED_OUTPUTS.get((days, miles, receipts), None)
        
        logging.info(f"Prediction request: days={days}, miles={miles}, receipts={receipts}")
        
        # Check if receipts end in .49 or .99 (CRITICAL PENALTY)
        receipt_cents = int(round((receipts % 1) * 100))
        has_49_99_penalty = receipt_cents in [49, 99]
        
        result = predict_reimbursement(days, miles, receipts)
        
        # Apply .49/.99 penalty if applicable
        if has_49_99_penalty:
            # Apply 10% penalty (optimal based on analysis)
            original_prediction = result
            result = result * (1 - 0.10)  # 10% reduction
            logging.info(f"*** APPLYING .49/.99 PENALTY ***")
            logging.info(f"Original prediction: ${original_prediction:.2f}")
            logging.info(f"With 10% penalty: ${result:.2f}")
            logging.info(f"Penalty amount: ${original_prediction - result:.2f}")
        
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