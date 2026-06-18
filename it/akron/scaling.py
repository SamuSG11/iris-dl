from flask import Blueprint, request, jsonify
import numpy as np
import pandas as pd

from src.scaling import FeatureScaler

scaling_bp = Blueprint("scaling", __name__)


@scaling_bp.route("/scale", methods=["POST"])
def scale_dataset():

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        X_train = data.get("X_train")
        X_val = data.get("X_val")
        X_test = data.get("X_test")
        y_train = data.get("y_train")
        y_val = data.get("y_val")
        y_test = data.get("y_test")

        if X_train is None or X_val is None or X_test is None:
            return jsonify({
                "error": "X_train, X_val, X_test, y_train, y_val, y_test are required"
            }), 400

        # 1. convert to DataFrame
        X_train = pd.DataFrame(X_train)
        X_val = pd.DataFrame(X_val)
        X_test = pd.DataFrame(X_test)
        y_train = pd.DataFrame(y_train)
        y_val = pd.DataFrame(y_val)
        y_test = pd.DataFrame(y_test)

        # 2. init scaler
        scaler = FeatureScaler()

        # 3. FIT SOLO SU TRAIN
        X_train_scaled = scaler.fit_transform(X_train)

        # 4. TRANSFORM SU VAL E TEST
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # 5. response
        return jsonify({
            "message": "Scaling completed successfully",
            "scaler_type": scaler.scaler_type,
            "X_train": X_train_scaled.values.tolist(),
            "X_val": X_val_scaled.values.tolist(),
            "X_test": X_test_scaled.values.tolist(),
            "y_train": y_train.values.tolist(),
            "y_val": y_val.values.tolist(),
            "y_test": y_test.values.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500