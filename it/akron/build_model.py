from flask import Blueprint, request, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

train_bp = Blueprint("train", __name__)


# 🔵 il model NON deve dipendere da X_train globale
def build_model(input_dim):
    model = Sequential([
        Dense(16, activation="relu", input_shape=(input_dim,)),
        Dense(8, activation="relu"),
        Dense(3, activation="softmax")
    ])
    return model


@train_bp.route("/train/adam", methods=["POST"])
def train_adam():

    # 🔥 QUI FAI IL data.get()
    data = request.get_json()

    X_train = np.array(data.get("X_train"))
    X_val = np.array(data.get("X_val"))
    X_test = np.array(data.get("X_test"))

    y_train = np.array(data.get("y_train"))
    y_val = np.array(data.get("y_val"))
    y_test = np.array(data.get("y_test"))

    # 🔴 controllo base
    if X_train is None or y_train is None:
        return jsonify({"error": "Missing training data"}), 400

    # 🔵 build model con input corretto
    model = build_model(X_train.shape[1])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 🔵 training
    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=16,
        validation_data=(X_val, y_val),
        verbose=0
    )

    # 🔵 test finale
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    return jsonify({
        "optimizer": "adam",
        "test_accuracy": float(acc),
        "test_loss": float(loss),
        "train_acc": float(history.history["accuracy"][-1]),
        "val_acc": float(history.history["val_accuracy"][-1])
    })