import pandas as pd
import numpy as np
from model import InsuranceFraudDetector
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze_data():
    """
    Load the insurance data with reduced memory usage.
    """
    print("Loading dataset...")
    # Read only necessary columns
    columns = ['AGE', 'POLICY_NUMBER', 'CLAIM_AMOUNT', 'PREMIUM_AMOUNT', 'CLAIM_STATUS', 'INCIDENT_SEVERITY']
    df = pd.read_csv('insurance_data.csv', usecols=columns)
    
    # Create a binary fraud indicator based on multiple factors
    df['fraud'] = (
        # High claim amount relative to premium
        (df['CLAIM_AMOUNT'] > df['PREMIUM_AMOUNT'] * 10) |
        # High incident severity
        (df['INCIDENT_SEVERITY'].str.contains('HIGH|CRITICAL', case=False, na=False)) |
        # Suspicious claim status
        (df['CLAIM_STATUS'].str.contains('SUSPICIOUS|FRAUD|REJECTED', case=False, na=False))
    ).astype(int)
    
    # Rename columns to match the model's expectations
    df = df.rename(columns={
        'AGE': 'age',
        'POLICY_NUMBER': 'policy_number',
        'CLAIM_AMOUNT': 'claim_amount',
        'PREMIUM_AMOUNT': 'income'  # Using premium amount as a proxy for income
    })
    
    print("\nDataset Overview:")
    print(f"Total records: {len(df)}")
    print(f"Features: {', '.join(df.columns)}")
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Print fraud distribution
    print("\nFraud Distribution:")
    print(df['fraud'].value_counts(normalize=True))
    
    return df

def train_and_evaluate_model(df):
    """
    Train the fraud detection model and evaluate its performance.
    """
    print("\nInitializing fraud detector...")
    fraud_detector = InsuranceFraudDetector()
    
    print("\nTraining model...")
    fraud_detector.train_model(df)
    
    # Save the trained model
    print("\nSaving model...")
    fraud_detector.save_model('trained_fraud_detector.joblib')
    
    return fraud_detector

def analyze_fraud_patterns(df, fraud_detector):
    """
    Analyze patterns in fraudulent claims with simplified visualization.
    """
    # Get fraud probabilities for all claims
    fraud_probabilities = fraud_detector.predict_fraud(df)
    
    # Analyze high-risk claims (probability > 0.7)
    high_risk_claims = df[fraud_probabilities > 0.7]
    
    print("\nHigh-Risk Claims Analysis:")
    print(f"Number of high-risk claims: {len(high_risk_claims)}")
    print(f"Percentage of high-risk claims: {(len(high_risk_claims) / len(df)) * 100:.2f}%")
    
    # Simplified visualization
    plt.figure(figsize=(8, 4))
    plt.hist(fraud_probabilities, bins=30, alpha=0.7)
    plt.title('Distribution of Fraud Probabilities')
    plt.xlabel('Fraud Probability')
    plt.ylabel('Count')
    plt.savefig('fraud_probability_distribution.png', dpi=100, bbox_inches='tight')
    plt.close()

def main():
    # Load and analyze the data
    df = load_and_analyze_data()
    
    # Train and evaluate the model
    fraud_detector = train_and_evaluate_model(df)
    
    # Analyze fraud patterns
    analyze_fraud_patterns(df, fraud_detector)
    
    print("\nFraud detection system training completed!")
    print("Model saved as 'trained_fraud_detector.joblib'")
    print("Fraud probability distribution plot saved as 'fraud_probability_distribution.png'")

if __name__ == '__main__':
    main() 