from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load the pre-trained Random Forest model
model = pickle.load(open('RandomForest.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Crop recommendation</p>"

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the request form
    N = float(request.form.get('N'))
    P = float(request.form.get('P'))
    K = float(request.form.get('K'))
    temperature = float(request.form.get('temperature'))
    humidity = float(request.form.get('humidity'))
    ph = float(request.form.get('ph'))
    rainfall = float(request.form.get('rainfall'))

    # Make prediction using the loaded model
    input_query = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_query)[0]

    # Prepare response JSON
    response = {'prediction': prediction}

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
