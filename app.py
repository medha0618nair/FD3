from flask import Flask, request, jsonify
from model import InsuranceFraudDetector
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
fraud_detector = InsuranceFraudDetector()
fraud_detector.load_model('trained_fraud_detector.joblib')

@app.route('/')
def home():
    return """
    <h1>Insurance Fraud Detection API</h1>
    <p>Send POST requests to /predict with JSON data containing insurance claim information.</p>
    <h2>Example Request:</h2>
    <pre>
    curl -X POST /predict \\
      -H "Content-Type: application/json" \\
      -d '{
        "age": 35,
        "income": 75000,
        "claim_amount": 15000,
        "policy_number": "12345"
      }'
    </pre>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided. Please send JSON data.'
            }), 400
        
        # Convert single claim to DataFrame
        df = pd.DataFrame([data])
        
        # Ensure all required columns are present
        required_columns = ['age', 'income', 'claim_amount', 'policy_number']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_columns)}'
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
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) 