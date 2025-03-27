import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

class InsuranceFraudDetector:
    def __init__(self):
        """
        Initialize the fraud detection system with key components.
        """
        self.model = None
        self.scaler = StandardScaler()
    
    def preprocess_data(self, df):
        """
        Preprocess the insurance claim data with simplified features.
        """
        # Handle missing values
        df = df.fillna({
            'age': df['age'].median(),
            'income': df['income'].median(),
            'claim_amount': df['claim_amount'].median()
        })
        
        # Simplified feature engineering
        df['income_per_claim'] = df['income'] / (df['claim_amount'] + 1)
        
        return df
    
    def extract_fraud_features(self, df):
        """
        Extract simplified fraud detection features.
        """
        features = [
            # Key fraud indicators
            df['claim_amount'] > df['claim_amount'].quantile(0.95),
            df['income_per_claim'] < df['income_per_claim'].quantile(0.1),
            df.groupby('policy_number')['claim_amount'].transform('count') > 2
        ]
        
        return np.column_stack([feat.astype(int) for feat in features])
    
    def train_model(self, df):
        """
        Train a simplified fraud detection model.
        """
        # Preprocess the data
        processed_df = self.preprocess_data(df)
        
        # Extract fraud features
        X_fraud_indicators = self.extract_fraud_features(processed_df)
        
        # Prepare features
        X = np.column_stack([
            processed_df[['age', 'income', 'claim_amount']].values,
            X_fraud_indicators
        ])
        y = processed_df['fraud']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train simplified Random Forest
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        print("Model Performance:")
        print(classification_report(y_test, y_pred))
    
    def predict_fraud(self, new_claims):
        """
        Predict fraud probability for new claims.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train_model() first.")
        
        processed_claims = self.preprocess_data(new_claims)
        X_fraud_indicators = self.extract_fraud_features(processed_claims)
        X = np.column_stack([
            processed_claims[['age', 'income', 'claim_amount']].values,
            X_fraud_indicators
        ])
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save_model(self, filepath='fraud_detection_model.joblib'):
        """
        Save the trained model and scaler.
        """
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, filepath, compress=3)
    
    def load_model(self, filepath='fraud_detection_model.joblib'):
        """
        Load a previously trained model and scaler.
        """
        saved_data = joblib.load(filepath)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']

# Example usage
def main():
    # Simulated insurance claims dataset
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 70, 1000),
        'gender': np.random.choice(['M', 'F'], 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'occupation': np.random.choice(['engineer', 'teacher', 'doctor', 'salesperson'], 1000),
        'claim_type': np.random.choice(['medical', 'car', 'home', 'life'], 1000),
        'claim_amount': np.random.exponential(5000, 1000),
        'policy_number': np.random.randint(10000, 99999, 1000),
        'fraud': np.random.choice([0, 1], 1000, p=[0.9, 0.1])  # 10% fraud rate
    }
    
    df = pd.DataFrame(data)
    
    # Initialize and train the fraud detector
    fraud_detector = InsuranceFraudDetector()
    fraud_detector.train_model(df)
    
    # Predict fraud probabilities for new claims
    new_claims = df.sample(10)  # Sample some claims for prediction
    fraud_probabilities = fraud_detector.predict_fraud(new_claims)
    
    # Print fraud probabilities
    print("\nFraud Probabilities for New Claims:")
    for prob in fraud_probabilities:
        print(f"{prob:.2%} probability of fraud")
    
    # Save the model for future use
    fraud_detector.save_model()

if __name__ == '__main__':
    main()