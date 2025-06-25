import pickle
from flask import Flask,request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
 
application = Flask(__name__) 
app = application

## import rigde regressor and standard scaler pickle 
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/getinfo_owner')
def getinfo_owner():
    return render_template('info_owner.html')

@app.route('/getinfo_dataset')
def getinfo_dataset():
    return render_template('info_dataset.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoints():
    if request.method == "POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        
        new_data_scaled=standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html',results=result[0])
    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")

# important commands that is written in the terminal
# 1 pip install -r requirements.txt
# 2 python application.py
# 3 ls -a 
# 4  git remote -v 
# 5 git remote rm origin
# 6 . git init
# 7  git add .
# 8  git status 
# 9  git commit -m "first commit"
# 10 git branch -M main
# 11 git remote add origin https://github.com/Amanarun2907/testforestfire.git
# 12 git push -u origin main
# 13 git add . 
# 14 git commit -m "first commit"
# 15 git push -u origin main