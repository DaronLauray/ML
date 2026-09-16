#!/usr/bin/env python
"""Quick model validation script - tests without full imports."""
import joblib
import os

MODEL_PATH = 'nba_winner_model.joblib'

try:
    payload = joblib.load(MODEL_PATH)
    print("✓ MODEL LOADED")
    print(f"  - Model type: {type(payload['model'])}")
    print(f"  - Features: {len(payload['features'])}")
    print(f"  - Feature names sample: {payload['features'][:3]}")
    print(f"✓ READY FOR PREDICTIONS")
except FileNotFoundError:
    print("✗ Model file not found")
except Exception as e:
    print(f"✗ Error: {e}")
