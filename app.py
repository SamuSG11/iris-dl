from flask import Flask

from it.akron.import_dataset import loader_bp
from it.akron.scaling import scaling_bp
from it.akron.splitter import splitter_bp
from it.akron.label_encoding import encoding_bp
from it.akron.build_model import train_bp


app = Flask(__name__)
app.register_blueprint(loader_bp)
app.register_blueprint(scaling_bp)
app.register_blueprint(splitter_bp)
app.register_blueprint(encoding_bp)
app.register_blueprint(train_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)