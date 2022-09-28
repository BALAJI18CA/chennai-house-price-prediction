from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict',methods=['POST','GET'])
def result():

    AREA= int(request.form['AREA'])
    INT_SQFT=int(request.form['INT_SQFT'])
    N_BEDROOM = int(request.form['N_BEDROOM'])
    N_BATHROOM= int(request.form['N_BATHROOM'])
    N_ROOM= int(request.form['N_ROOM'])
    SALE_COND= int(request.form['SALE_COND'])
    PARK_FACIL= int(request.form['PARK_FACIL'])
    UTILITY_AVAIL= int(request.form['UTILITY_AVAIL'])
    STREET= int(request.form['STREET'])
    MZZONE= int(request.form['MZZONE'])
    HOUSE_AGE= int(request.form['HOUSE_AGE'])
    BUILDTYPE= int(request.form['BUILDTYPE'])
   

    X= np.array([[AREA,INT_SQFT,N_BEDROOM,N_BATHROOM,N_ROOM,SALE_COND,PARK_FACIL,UTILITY_AVAIL,STREET,MZZONE,HOUSE_AGE,BUILDTYPE]])

    scaler_path= r'D:\chennai house price\model\gredient_model.sev'

    sc=joblib.load(scaler_path)

    X_std= sc.transform(X)

    model_path=r'D:\chennai house price\model\gredient_model.sev'

    model= joblib.load(model_path)

    Y_pred=model.predict(X_std)

    return jsonify({'Prediction': float(Y_pred)})

if __name__ == "__main__":
    app.run(debug=True, port=9457)