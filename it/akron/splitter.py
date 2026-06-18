from flask import Flask, Blueprint, request, jsonify
import numpy as np

from src.splitter import DatasetSplitter

splitter_bp = Blueprint("splitter", __name__)

@splitter_bp.route("/split", methods=["POST"])
def split_dataset():

    data = request.get_json()

    print("JSON ricevuto:")
    print(data)

    print("X presente?", "X" in data if data else False)
    print("y presente?", "y" in data if data else False)

    # 1. prendo X e y dal JSON
    X = np.array(data.get("X"))
    y = data.get("y_encoded")
    y = np.array(y)

    # 2. controllo base
    if X is None or y is None:
        return jsonify({"error": "X e y sono obbligatori"}), 400

    print("X:", type(X), X is None)
    print("y:", type(y), y is None)
    print("X shape:", getattr(X, "shape", None))
    print("y shape:", getattr(y, "shape", None))

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