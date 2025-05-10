from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input', methods=['GET', 'POST'])
def input_form():
    if request.method == 'POST':
        # Get user info from welcome page and pass to input page
        name = request.form['name']
        email = request.form['email']
        return render_template('input.html', name=name, email=email)
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    age = float(request.form['age'])
    sex = int(request.form['sex'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = int(request.form['smoker'])
    region = int(request.form['region'])
    
    # Prepare input for model
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Get user info to display
    name = request.form.get('name', 'User')
    
    return render_template('output.html', prediction=prediction, name=name)

if __name__ == '__main__':
    app.run(debug=True)