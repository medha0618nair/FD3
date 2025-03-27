from flask import Flask, request, jsonify
from model import InsuranceFraudDetector
import pandas as pd

app = Flask(__name__)

# Load the trained model
fraud_detector = InsuranceFraudDetector()
fraud_detector.load_model('trained_fraud_detector.joblib')

@app.route('/')
def home():
    return """
    <h1>Insurance Fraud Detection API</h1>
    <p>Send POST requests to /predict with JSON data containing insurance claim information.</p>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert single claim to DataFrame
        df = pd.DataFrame([data])
        
        # Ensure all required columns are present
        required_columns = ['age', 'income', 'claim_amount', 'policy_number']
        for col in required_columns:
            if col not in df.columns:
                return jsonify({
                    'error': f'Missing required field: {col}'
                }), 400
        
        # Make prediction
        fraud_probability = fraud_detector.predict_fraud(df)[0]
        
        # Return prediction
        return jsonify({
            'fraud_probability': float(fraud_probability),
            'is_high_risk': bool(fraud_probability > 0.7)
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000) 