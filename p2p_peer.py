from fl_peer import FLPeer
from MLTSModel import MLTSModel
from csv_ts_data_provider import CSVTSDataProvider
import torch

from threading import Thread
from flask import Flask, send_file

from requests import post, get
from shutil import copyfileobj

from time import sleep

app = Flask(__name__)
trained = False
peer_id = 0

# Download model from another peer
@app.route('/latest_model', methods=['GET'])
def download_model():
    try:
        return send_file(f'temp/{peer_id}_model.pth')
    except FileNotFoundError:
        return 'File not found', 404

# Send a newly trained model to a peer to trigger aggregation
@app.route('/ready', methods=['GET'])
def model_ready():
    if trained is True:
        return 'yes', 200
    else:
        return 'no', 200

class P2PPeer(FLPeer):
    def __init__(self, REGISTRATION_ADDRESS, LOCAL_PORT, ml_model: MLTSModel, data_provider: CSVTSDataProvider):
        FLPeer.__init__(self, ml_model, data_provider)
        self.LOCAL_PORT = LOCAL_PORT
        self.REGISTRATION_ADDRESS = REGISTRATION_ADDRESS

    def start(self):
        print('Registering self with registration node.')
        self.register_with_registration()

        print('Starting Flask app for file requests')
        self.flask_thread = Thread(target=self.start_flask_app)
        self.flask_thread.start()

    def register_with_registration(self):
        returned_peer_id = int(post(self.REGISTRATION_ADDRESS + '/register', json={'ext_port': str(self.LOCAL_PORT)}).text)
        print('Registered with registration node.')

        global peer_id
        peer_id = returned_peer_id

    def start_flask_app(self):
        app.run(host="0.0.0.0", port=self.LOCAL_PORT)

    def get_peer_list(self):
        return get(self.REGISTRATION_ADDRESS + '/peers').json()['peers']

    def fetch(self, url, target_filename):
        with get(url, stream=True) as request:
            with open(target_filename, 'wb') as file:
                copyfileobj(request.raw, file)

    def train(self):
        FLPeer.train(self)
        self.ml_model.save(f'temp/{peer_id}_model.pth')

        global trained
        trained = True

    def wait_for_other_peers(self):        
        while True:
            other_peers_ready = True
            sleep(15)
            
            for peer in self.get_peer_list():
                url = 'http://' + peer['address'] + ':' + peer['ext_port'] + '/ready'

                response = get(url).text
                
                if response == 'no':
                    other_peers_ready = False
                    break

            if other_peers_ready:
                break

    def aggregate(self):
        peer_models = []
        
        for peer in self.get_peer_list():
            peer_index = peer['peer_index']
            self.fetch('http://' + peer['address'] + ':' + peer['ext_port'] + '/latest_model', f'temp/peer_{peer_id}_{peer_index}_model.pth')
            peer_models.append(f'temp/peer_{peer_id}_{peer_index}_model.pth')

        print(peer_models)
        FLPeer.aggregate(self, peer_models)