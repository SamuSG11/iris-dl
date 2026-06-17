from flask import Blueprint, request, jsonify

from src import DataLoader

loader_bp = Blueprint("loader", __name__)

@loader_bp.route("/load", methods=["POST"])
def load_dataset():
    """
    Carica un dataset e crea una nuova pipeline.

    Body JSON:
    {
        "file_path": "iris_nuovo.CSV"
    }
    """

    try:

        data = request.get_json()

        if not data:
            return jsonify(
                {"error": "Missing JSON body"}
            ), 400

        file_path = data.get("file_path")

        if not file_path:
            return jsonify(
                {"error": "file_path is required"}
            ), 400

        loader = DataLoader()

        X = loader.load_csv(file_path)


        return jsonify(
            {
                "message": "Dataset loaded successfully",
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "features": list(X.columns),
            }
        )

    except Exception as e:

        return jsonify(
            {"error": str(e)}
        ), 500
