from flask import Flask, request, jsonify
from model import InsuranceFraudDetector
import pandas as pd
import os
import logging
from flask_swagger_ui import get_swaggerui_blueprint

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
SWAGGER_URL = '/docs'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Insurance Fraud Detection API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Load the trained model
logger.info("Loading the fraud detection model...")
fraud_detector = InsuranceFraudDetector()
fraud_detector.load_model('trained_fraud_detector.joblib')
logger.info("Model loaded successfully!")

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
    <p>Status: API is running</p>
    <p>API Documentation: <a href="/docs">/docs</a></p>
    """

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict fraud probability for an insurance claim
    ---
    tags:
      - Predictions
    parameters:
      - in: body
        name: body
        schema:
          type: object
          required:
            - age
            - income
            - claim_amount
            - policy_number
          properties:
            age:
              type: integer
              description: Age of the policyholder
            income:
              type: number
              description: Annual income of the policyholder
            claim_amount:
              type: number
              description: Amount being claimed
            policy_number:
              type: string
              description: Unique policy number
    responses:
      200:
        description: Successful prediction
        schema:
          type: object
          properties:
            fraud_probability:
              type: number
            is_high_risk:
              type: boolean
      400:
        description: Invalid input
      500:
        description: Server error
    """
    try:
        logger.info("Received prediction request")
        # Get JSON data from request
        data = request.get_json()
        logger.info(f"Request data: {data}")
        
        if not data:
            logger.error("No data provided in request")
            return jsonify({
                'error': 'No data provided. Please send JSON data.'
            }), 400
        
        # Convert single claim to DataFrame
        df = pd.DataFrame([data])
        logger.info(f"Created DataFrame with columns: {df.columns.tolist()}")
        
        # Ensure all required columns are present
        required_columns = ['age', 'income', 'claim_amount', 'policy_number']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_columns)}'
            }), 400
        
        # Make prediction
        logger.info("Making prediction...")
        fraud_probability = fraud_detector.predict_fraud(df)[0]
        logger.info(f"Prediction completed. Fraud probability: {fraud_probability}")
        
        # Return prediction
        response = {
            'fraud_probability': float(fraud_probability),
            'is_high_risk': bool(fraud_probability > 0.7)
        }
        logger.info(f"Sending response: {response}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) 