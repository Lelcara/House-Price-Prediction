#%%
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model, scaler, selector, and selected feature names
with open('house_price_prediction.pkl', 'rb') as f:
    model, scaler, selector, selected_features = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect only the selected features in the correct order
        input_data = {}
        for feature in selected_features:
            input_data[feature] = float(request.form[feature])
        
        # Create DataFrame for consistent formatting
        input_df = pd.DataFrame([input_data])

        # Fill missing features with zeros (if any, for scaler input)
        all_features = scaler.feature_names_in_
        full_input = pd.DataFrame([[0]*len(all_features)], columns=all_features)
        for key in input_data:
            full_input[key] = input_data[key]

        # Scale and select
        scaled_input = scaler.transform(full_input)
        selected_input = selector.transform(scaled_input)

        # Predict
        prediction = model.predict(selected_input)[0]
        output = round(prediction, 2)

        return render_template('index.html', prediction_text=f"Predicted Price: ${output}")
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

# %%
