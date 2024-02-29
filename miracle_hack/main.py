from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pickle
import json

# importing model
with open('static/d.json', 'r') as file:
    fertilizer_data = json.load(file)
model = pickle.load(open('models/model.pkl', 'rb'))
model1 = pickle.load(open('models/standscaler.pkl', 'rb'))
model2 = pickle.load(open('models/minmaxscaler.pkl', 'rb'))

# Dummy user data (replace with a database for real-world applications)
user_credentials = {'user1': 'password1'}

# creating flask app
app = Flask(__name__)

# register route
@app.route('/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username not in user_credentials:
            user_credentials[username] = password
            return redirect(url_for('login'))
        else:
            return render_template('register.html', message='Username already taken. Please choose a different one.')
    else:
        return render_template('register.html')

#login
@app.route('/register/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if check_credentials(username, password):
            return redirect(url_for('index'))
        else:
            return render_template('login.html', message='Invalid credentials. Please try again.')
    else:
        return render_template('login.html')

def check_credentials(username, password):
    return username in user_credentials and user_credentials[username] == password

# Main routes
@app.route('/register/login/index')
def index():
    return render_template("index.html")

@app.route('/about_us')
def about():
    return render_template('about_us.html')

@app.route('/overview', methods=['GET'])
def overview():
    return render_template('overview.html')

@app.route('/predict', methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])
    K = int(request.form['Potassium'])
    temp =int( request.form['Temperature'])
    humidity =int( request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = int(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = model2.transform(single_pred)
    final_features = model1.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop_name = crop_dict[prediction[0]]
        fertilizer_info = fertilizer_data["crops"].get(crop_name, {}).get("fertilizers", {})
    
        if fertilizer_info:
          result = "{} is the best crop to be cultivated right there. \n\nRecommended fertilizers:\nNitrogen: {}, \n\nPhosphorus: {}, \n\nPotassium: {}".format(
            crop_name, fertilizer_info.get("nitrogen", ""), fertilizer_info.get("phosphorus", ""), fertilizer_info.get("potassium", "")
        )
        else:
            result = "{} is the best crop to be cultivated right there. Unfortunately, we don't have specific fertilizer information for this crop.".format(crop_name)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    return render_template('index.html', result=result)
# python main
if __name__ == "__main__":
    app.run(use_reloader=True,debug=True,port=3002)