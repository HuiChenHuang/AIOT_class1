from flask import Flask, request, render_template
import joblib
import pandas as pd

# Load the model
model = joblib.load('linear_regression_model.pkl')

# Create a Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the form
    features = [float(request.form['feature1']),
                float(request.form['feature2']),
                float(request.form['feature3']),
                float(request.form['feature4']),
                float(request.form['feature5'])]  # Add more if needed
    
    # Convert features into a DataFrame for the model
    input_data = pd.DataFrame([features], columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    return render_template('index.html', prediction_text=f'Predicted Value: {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)
