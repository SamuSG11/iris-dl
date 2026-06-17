from flask import Blueprint, request, jsonify
import numpy as np

from src.encoding import TargetOHE

encoding_bp = Blueprint("encoding", __name__)

@encoding_bp.route("/encode", methods=["POST"])
def one_hot_encode_target():

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        y = data.get("y")

        if y is None:
            return jsonify({"error": "y is required"}), 400

        y = np.array(y)

        # 1. OHE
        encoder = TargetOHE()
        y_encoded = encoder.fit_transform(y)

        # 2. risposta
        return jsonify({
            "classes": encoder.classes_.tolist(),
            "y_encoded": y_encoded.tolist(),
            "shape": list(y_encoded.shape)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500