import pandas as pd
import os

# Set file path
data_path = '/home/UserData/ljx/beidabingli/队列符合277.xlsx'

# Check if file exists
if not os.path.exists(data_path):
    print(f"File not found: {data_path}")
    exit(1)

# Load data
try:
    df = pd.read_excel(data_path)
    print("Data loaded successfully.")
    print(f"Shape: {df.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Display basic info
print("\n--- Columns ---")
print(df.columns.tolist())

print("\n--- Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Head ---")
print(df.head())
