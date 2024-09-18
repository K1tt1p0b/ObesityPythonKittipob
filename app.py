from flask import Flask, request, jsonify
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model = joblib.load("obesity_model.pkl")

@app.route('/api/obesity', methods=['POST'])
def predict_obesity():
    
        # Get input data from the form
        age = int(request.form.get('age', 0))  # Default to 0 if not provided
        gender = int(request.form.get('gender', 0))  # Default to 0 if not provided
        height = float(request.form.get('height', 0.0))  # Default to 0.0 if not provided
        weight = float(request.form.get('weight', 0.0))  # Default to 0.0 if not provided
        bmi = float(request.form.get('bmi', 0.0))  # Default to 0.0 if not provided
        physical_activity_level = int(request.form.get('physical_activity_level', 1))  # Default to 0 if not provided

        # Prepare the input for the model
        x = np.array([[age, gender, height, weight, bmi, physical_activity_level]])

        # Predict using the model
        prediction = model.predict(x)

        # Return the result as a response
        return jsonify({'obesity_risk': prediction[0].tolist()}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
