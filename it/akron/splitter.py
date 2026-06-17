from flask import Flask, request, jsonify
import numpy as np

from src.splitter import DatasetSplitter

splitter_bp = Blueprint("splitter", __name__)

@app.route("/split", methods=["POST"])
def split_dataset():

    data = request.get_json()

    # 1. prendo X e y dal JSON
    X = np.array(data.get("X"))
    y = np.array(data.get("y"))

    # 2. controllo base
    if X is None or y is None:
        return jsonify({"error": "X e y sono obbligatori"}), 400

    # 3. split
    splitter = DatasetSplitter()
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)

    # 4. risposta JSON-safe
    response = {
        "X_train": X_train.tolist(),
        "X_val": X_val.tolist(),
        "X_test": X_test.tolist(),
        "y_train": y_train.tolist(),
        "y_val": y_val.tolist(),
        "y_test": y_test.tolist(),
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)