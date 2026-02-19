from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath',
            'TotalBsmtSF', 'GarageCars', 'YearBuilt', 'OverallQual',
            'LotArea', 'Fireplaces']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    values = [float(request.form[f]) for f in features]
    scaled = scaler.transform([values])
    log_pred = model.predict(scaled)[0]
    price = np.expm1(log_pred)  # Reverse log transform
    formatted = f"{price:,.0f}"
    return render_template('index.html', prediction=formatted)

if __name__ == '__main__':
    app.run(debug=True)
