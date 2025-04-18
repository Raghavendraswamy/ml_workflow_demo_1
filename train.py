# train.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

# Load Data
df = pd.read_csv('data/data.csv')

# Simple example: predict y from x
X = df[['x']]
y = df['y']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
os.makedirs('model', exist_ok=True)
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved.")
