from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the pre-trained Random Forest model
model = pickle.load(open('RandomForest.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

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

    # Render the prediction result template and pass the prediction result and user inputs
    return render_template('prediction_result.html', label=prediction, N=N, P=P, K=K, temperature=temperature,
                           humidity=humidity, ph=ph, rainfall=rainfall)

if __name__ == "__main__":
    app.run(debug=True)
