from flask import Blueprint, request, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

train_bp = Blueprint("train", __name__)


def build_model(input_dim):
    model = Sequential([
        Dense(16, activation="relu", input_shape=(input_dim,)),
        Dense(8, activation="relu"),
        Dense(3, activation="softmax")
    ])
    return model


def safe_array(data, key):
    """
    Converte in numpy array in modo sicuro.
    Lancia errore chiaro se manca la chiave o è vuota.
    """
    if data is None:
        raise ValueError("JSON body is missing")

    if key not in data:
        raise KeyError(f"Missing key: {key}")

    value = data.get(key)

    if value is None:
        raise ValueError(f"{key} is None")

    arr = np.array(value, dtype=np.float32)

    if arr.size == 0:
        raise ValueError(f"{key} is empty")

    return arr


@train_bp.route("/train/adam", methods=["POST"])
def train_adam():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        # load dataset in modo sicuro
        X_train = safe_array(data, "X_train")
        X_val   = safe_array(data, "X_val")
        X_test  = safe_array(data, "X_test")

        y_train = safe_array(data, "y_train")
        y_val   = safe_array(data, "y_val")
        y_test  = safe_array(data, "y_test")

        # sanity check shapes
        if X_train.shape[0] != y_train.shape[0]:
            return jsonify({"error": "X_train and y_train size mismatch"}), 400

        # model
        model = build_model(X_train.shape[1])

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        # train
        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=16,
            validation_data=(X_val, y_val),
            verbose=0,
            callbacks=[early_stop]
        )

        # evaluate
        loss, acc = model.evaluate(X_test, y_test, verbose=0)

        best_epoch = np.argmin(history.history["val_loss"])

        return jsonify({
            "optimizer": "adam",
            "epochs_run": len(history.history["loss"]),
            "best_epoch": int(best_epoch+1),
            "test_accuracy": float(acc),
            "test_loss": float(loss),
            "train_acc": float(history.history["accuracy"][best_epoch]),
            "val_acc": float(history.history["val_accuracy"][best_epoch]),
            "overfitting_gap": float(
                history.history["accuracy"][best_epoch] - history.history["val_accuracy"][best_epoch]   
            )
        })

    except KeyError as e:
        return jsonify({
            "error": str(e),
            "type": "KeyError"
        }), 400

    except ValueError as e:
        return jsonify({
            "error": str(e),
            "type": "ValueError"
        }), 400

    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500