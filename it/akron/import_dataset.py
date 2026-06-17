from flask import Blueprint, request, jsonify

from src import DataLoader

loader_bp = Blueprint("loader", __name__)


@loader_bp.route("/load", methods=["POST"])
def load_dataset():
    """
    Carica un dataset e separa feature e target.

    Body JSON:
    {
        "file_path": "iris.csv",
        "target": "Species"
    }
    """

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        file_path = data.get("file_path")
        target_col = data.get("target")

        if not file_path:
            return jsonify({"error": "file_path is required"}), 400

        if not target_col:
            return jsonify({"error": "target column is required"}), 400

        # Load dataset
        loader = DataLoader()
        df = loader.load_csv(file_path)

        if df is None:
            return jsonify({"error": "Failed to load dataset"}), 500

        # Split features and target
        if target_col not in df.columns:
            return jsonify({
                "error": f"Target column '{target_col}' not found in dataset",
                "available_columns": list(df.columns)
            }), 400

        X = df.drop(columns=[target_col])
        y = df[target_col]

        return jsonify({
            "message": "Dataset loaded successfully",
            "n_samples": df.shape[0],
            "n_features": X.shape[1],
            "features": list(X.columns),
            "target": target_col
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500