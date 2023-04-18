from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle as pick


app = Flask(__name__)

@app.route('/api/', methods=['POST'])
def predict():
    """
    Performs a classification over models inputed in pickle.
    :param: None
    :return: The prediction in JSON format
    """
    raw_data: pd.DataFrame = request.get_json()

    # transformations
    data: np.array = imputer.transform(raw_data)
    data = normalizer.transform(data)

    #predict using majority voting between the three models
    votes = model_rf.predict(data) + model_svc.predict(data) + model_lgrg.predict(data)
    prediction = np.array2string(1*(votes>1))

    return jsonify(prediction)

if __name__ == '__main__':
    imputer = pick.load(open("imputer.pickle", "rb"))
    model_rf = pick.load(open("model_rf.pickle","rb"))
    model_svc = pick.load(open("model_svc.pickle","rb"))
    model_lgrg = pick.load(open("model_lgrg.pickle","rb"))
    normalizer = pick.load(open("normalizer.pickle","rb"))

    app.run(debug=True, host='0.0.0.0')
