from flask import Flask, request, jsonify, render_template, send_file
import joblib
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3600  # Cache control

# Load the trained model
model_path = "heart_disease_model.pkl"
model = None
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully at {datetime.now()}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

# Optional: Serve styles.css from root if needed
@app.route('/styles.css')
def serve_styles():
    return send_file(os.path.join(app.root_path, 'static', 'styles.css'))

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('index.html', 
                               prediction="Model Not Available",
                               result_class='h2-fail',
                               show_modal=True)
    try:
        # Extract and validate features
        feature_keys = ["age", "sex", "cp", "trestbps", "chol", "fbs", 
                        "restecg", "thalach", "exang", "oldpeak", "slope", 
                        "ca", "thal"]
        
        features = []
        for key in feature_keys:
            value = request.form.get(key, '')
            if not value:
                return render_template('index.html',
                                       prediction="Missing Fields",
                                       result_class='h2-fail')
            try:
                features.append(float(value))
            except ValueError:
                return render_template('index.html',
                                       prediction="Invalid Input",
                                       result_class='h2-fail')

        # Make prediction
        features_array = np.array([features])
        prediction = model.predict(features_array)[0]
        
        result = 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'
        result_class = 'h2-fail' if prediction == 1 else 'h2-success'
        
        return render_template('index.html', 
                               prediction=result, 
                               result_class=result_class,
                               show_modal=True)

    except Exception as e:
        return render_template('index.html',
                               prediction=f"Error: {str(e)}",
                               result_class='h2-fail',
                               show_modal=True)

@app.after_request
def add_header(response):
    # Disable caching for prediction results
    if request.path == '/predict':
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
