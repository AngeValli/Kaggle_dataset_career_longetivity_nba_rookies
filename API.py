from flask import Flask, request, jsonify
from typing import List
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
    raw_data: List[float] = request.get_json()

    # verify data inputs
    assert all((isinstance(x, float) for x in l) for l in raw_data)

    # transformations
    data: np.array = imputer.transform(raw_data)
    preds: np.array = k_means_labeling.predict(data)
    preds = preds.reshape(len(preds), 1)
    data = np.append(data, preds, axis=1)
    data = normalizer.transform(data)

    # predict using majority voting between the three models
    votes = model_rf.predict(data) + model_svc.predict(data) + model_lgrg.predict(data)
    prediction = np.array2string(1*(votes>1))

    return jsonify(prediction)

if __name__ == '__main__':
    imputer: pick = pick.load(open(c.IMPUTER, "rb"))
    k_means_labeling: pick = pick.load(open(c.KMEANS_MODEL, "rb"))
    model_rf: pick = pick.load(open(c.MODEL_RF,"rb"))
    model_svc: pick = pick.load(open(c.MODEL_SVC,"rb"))
    model_lgrg: pick = pick.load(open(c.MODEL_LGRG,"rb"))
    normalizer: pick = pick.load(open(c.NORMALIZER,"rb"))

    app.run(debug=True, host='0.0.0.0')
