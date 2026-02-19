import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
df = pd.read_csv('data/house_data.csv')

# Select important features (keeping it simple but professional)
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath',
            'TotalBsmtSF', 'GarageCars', 'YearBuilt', 'OverallQual',
            'LotArea', 'Fireplaces']

target = 'SalePrice'

# Drop rows with missing values in selected columns
df = df[features + [target]].dropna()

X = df[features]
y = np.log1p(df[target])  # Log transform to handle skewness

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
preds = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f" Model Trained Successfully!")
print(f" RMSE: {rmse:.4f}")
print(f" R² Score: {r2:.4f}")

# Save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print(" Model saved as model.pkl")

