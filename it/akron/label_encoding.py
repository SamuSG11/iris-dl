from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd
from src.encoding import TargetOHE

encoding_bp = Blueprint("encoding", __name__)


@encoding_bp.route("/encode", methods=["POST"])
def one_hot_encode_target():

    data = request.get_json()

    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    X = data.get("X")
    y = data.get("y")

    if X is None or y is None:
        return jsonify({"error": "X and y are required"}), 400

    X = np.array(X)
    y = np.array(y)

    # encoding
    encoder = TargetOHE()
    y_encoded = encoder.fit_transform(y)

    # stampa / return dataset completo
    return jsonify({
        "X": X.tolist(),
        "y_encoded": y_encoded.tolist(),
        "classes": encoder.classes_.tolist()
    })