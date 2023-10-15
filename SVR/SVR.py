from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd



# Load the dataset
df = pd.read_csv('SVR/SVR.py')

# Show the first few rows to understand what the data looks like
print(df.head())

# Assume df is a pandas DataFrame containing your housing data
# and 'Price' is the column you're trying to predict

# Feature Selection
X = df.drop('Price', axis=1)
y = df['Price']

# Data Standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr.fit(X_train, y_train)

# Prediction
y_pred = svr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
