from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
app=Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)

@app.route("/predict_disease",methods= ["POST"])
def predict_disease():
    try:
        data = request.get_json()
        age=data.get("age")
        gender=data.get("gender")
        height=data.get("height")
        weight=data.get("weight")
        ap_hi=data.get("ap_hi")
        ap_lo=data.get("ap_lo")
        smoke=data.get("smoke")
        alco=data.get("alco")
        new_data = [[age, gender, height, weight, 100, 60, smoke, alco]]
        #load the model and predict the data
        #create a new data
        # model = joblib.load(open("../cardiovascular_prediction/model/cardiovascular_prediction_model.plk", 'rb'))
        # scaler = joblib.load(open("../cardiovascular_prediction/model/cardiovascular_prediction_scaler.plk", 'rb'))
        model = joblib.load(open("models/cardiovascular_prediction_model.plk", 'rb'))
        scaler = joblib.load(open("models/cardiovascular_prediction_scaler.plk", 'rb'))
        # model = pickle.load(open("../model/cardiovascular_prediction_model.plk", 'rb'))
        age =age*365
        # new_data = [[7474, 1, 156, 56, 100, 60, 0, 0]]
        new_data = [[age, gender, height, weight, ap_hi, ap_lo, smoke, alco]]
        print(new_data)
        new_data = scaler.transform(new_data)
        result = model.predict(new_data)
        print(jsonify({"status":1,'prediction': result.tolist()[0]}))
        return jsonify({"status":1,'prediction': result.tolist()[0]})
    except Exception as e:
        return jsonify({"status":0,"error":str(e)})

@app.route("/",methods=["GET"])
def initial_route():
    return "Hello world";

if __name__ == '__main__':
    app.run(debug=True  )

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
