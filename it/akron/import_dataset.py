from flask import Blueprint, request, jsonify
import uuid

from src import DataLoader

loader_bp = Blueprint("loader", __name__)

@loader_bp.route("/load", methods=["POST"])
def load_dataset():
    """
    Carica un dataset e crea una nuova pipeline.

    Body JSON:
    {
        "file_path": "data/seeds_dataset.txt"
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

        X, y = loader.load_seeds_dataset(file_path)

        pipeline_id = str(uuid.uuid4())

        return jsonify(
            {
                "message": "Dataset loaded successfully",
                "pipeline_id": pipeline_id,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "features": list(X.columns),
                "target": y.columns[0]
            }
        )

    except Exception as e:

        return jsonify(
            {"error": str(e)}
        ), 500
