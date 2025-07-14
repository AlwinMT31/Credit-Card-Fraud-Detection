import joblib

# Save model and scaler
joblib.dump(xgb_model, 'random_forest_fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
