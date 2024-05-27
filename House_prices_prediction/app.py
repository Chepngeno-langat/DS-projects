from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy as np
import pandas as pd
import math

app = Flask(__name__)

model = joblib.load('gb_model.pkl')
model_columns = joblib.load('model_columns.pkl')
one_hot_encoder = joblib.load('one_hot_encoder.pkl')

categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Extract features from JSON data
    input_features = data['features']

    # Create a DataFrame from the input features
    input_data = pd.DataFrame([input_features])

    # Separate numerical and categorical data
    numerical_data = input_data[numerical_features]
    categorical_data = input_data[categorical_features]

    # Encode the categorical data
    encoded_categorical_data = one_hot_encoder.transform(categorical_data)

    # Create a DataFrame with the encoded categorical data
    encoded_categorical_df = pd.DataFrame(encoded_categorical_data,
                                          columns=one_hot_encoder.get_feature_names_out(categorical_features))

    # # Combine numerical and encoded categorical data
    final_input_data = pd.concat([numerical_data.reset_index(drop=True), encoded_categorical_df.reset_index(drop=True)],
                                 axis=1)

    # Make a prediction using the loaded model
    prediction = model.predict(final_input_data)

    return jsonify({'prediction': math.ceil(float(prediction[0]))})

if __name__ == '__main__':
    app.run(debug=True)

