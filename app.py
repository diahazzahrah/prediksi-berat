from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def prediksi_berat():
    if request.method == 'GET':
        return render_template("prediksi-berat.html")
    elif request.method == 'POST':
        print(dict(request.form))
        gender_features = list(dict(request.form).values())
        if gender_features[1] == "Male":
            gender_features[1] = 0
        else: 
            gender_features[1] = 1
        gender_features = np.array([float(x) for x in gender_features])
        model = joblib.load("model-development/model-prediksi.pkl")
        gender_features = pd.DataFrame({"gender":gender_features[1], "tinggi":gender_features[0]}, index=[0])
        print(gender_features)
        

        result = model.predict(gender_features)
      
        return render_template('prediksi-berat.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)