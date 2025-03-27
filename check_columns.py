import pandas as pd

print("Reading CSV file...")
df = pd.read_csv('insurance_data.csv')
print("\nColumns in the CSV file:")
print(df.columns.tolist())
print("\nFirst few rows of the data:")
print(df.head()) 