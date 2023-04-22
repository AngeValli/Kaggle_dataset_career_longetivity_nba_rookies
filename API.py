from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle as pick

import constants as c


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
    imputer: pick = pick.load(open(c.IMPUTER, "rb"))
    model_rf: pick = pick.load(open(c.MODEL_RF,"rb"))
    model_svc: pick = pick.load(open(c.MODEL_SVC,"rb"))
    model_lgrg: pick = pick.load(open(c.MODEL_LGRG,"rb"))
    normalizer: pick = pick.load(open(c.NORMALIZER,"rb"))

    app.run(debug=True, host='0.0.0.0')
