from flask import Blueprint

home_bp = Blueprint("home", __name__)


@home_bp.route("/", methods=["GET"])
def intro():
    return """
        <h1>--- BENVENUTO NELL'API RETI NEURALI ---</h1>

        <p>
            REST API per l'analisi del dataset Iris e la divisione
            dei vari semi in clusters attraverso l'uso delle RN
        </p>

        <h2>CARICAMENTO DEL DATASET:</h2>

        <h3>Dataset load</h3>
        <ul>
            <li>POST /load
            <li> inserire come body il json:</li>
            <li> {"file_path": "iris_nuovo.csv", "target": "Species"}</li>
        </ul>

        <h2>PULIZIA DEL DATASET:</h2>

        <h3>Pulizia del dataset in ordine</h3>
        <ul>
            <li>POST /encode</li>
            <li>POST /split</li>
            <li>POST /scale</li>
            <li></li>
            <li>in ogni endpoint inserire l'output di quello prima</li>
        </ul>
        
        <h2>RETE NEURALE:</h2>

        <h3>build model h1=16 e h2=8 e settata su Adam con 30 epochs</h3>
        <ul>
            <li>POST /train/adam</li>
            <li>passandogli in input il risultato di scale</li>
        </ul>

        """