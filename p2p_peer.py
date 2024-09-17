from fl_peer import FLPeer
from MLTSModel import MLTSModel
from csv_ts_data_provider import CSVTSDataProvider
import torch

from threading import Thread
from flask import Flask, send_file, request

from requests import post, get
from shutil import copyfileobj

from time import sleep

import nacl.utils
from nacl.public import PrivateKey, PublicKey, Box
from nacl.encoding import HexEncoder

from io import BytesIO

app = Flask(__name__)
trained = False
peer_id = 0
peer_inst = None

# Download model from another peer
@app.route('/latest_model', methods=['GET'])
def download_model():
    print('Key {}', request.args.get('key', type=str))
    requester_public_key = PublicKey(bytes.fromhex(request.args.get('key', type=str)), HexEncoder)

    try:
        buffer = BytesIO()
        
        with open(f'temp/{peer_id}_model.pth', 'rb') as f:
            buffer.write(peer_inst.encrypt(requester_public_key, f.read()))
        
        buffer.seek(0)

        return send_file(buffer, download_name=f'{peer_id}_model.pth', mimetype='application/octet-stream')
    except FileNotFoundError:
        return 'File not found', 404

@app.route('/key', methods=['GET'])
def get_key():
    return peer_inst.public_key.encode(HexEncoder).hex()

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
        self.secret_key = PrivateKey.generate()
        self.public_key = self.secret_key.public_key

        print(self.public_key.encode(HexEncoder).hex())

        global peer_inst
        peer_inst = self

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

    def encrypt(self, target_public_key, plaintext: bytes):
        box = Box(self.secret_key, target_public_key)
        return box.encrypt(plaintext)
    
    def decrypt(self, target_public_key, ciphertext: bytes):
        box = Box(self.secret_key, target_public_key)
        return box.decrypt(ciphertext)

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
        self_public_key_string = self.public_key.encode(HexEncoder).hex()
        peer_models = []
        
        for peer in self.get_peer_list():
            peer_index = peer['peer_index']
            peer_public_key = PublicKey(bytes.fromhex(get(f'http://' + peer['address'] + ':' + peer['ext_port'] + '/key').text), HexEncoder)
            self.fetch('http://' + peer['address'] + ':' + peer['ext_port'] + '/latest_model?key=' + self_public_key_string, f'temp/peer_{peer_id}_{peer_index}_model_enc.pth')

            with open(f'temp/peer_{peer_id}_{peer_index}_model_enc.pth', 'rb') as in_f, open(f'temp/peer_{peer_id}_{peer_index}_model.pth', 'wb') as out_f:
                out_f.write(self.decrypt(peer_public_key, in_f.read()))

            peer_models.append(f'temp/peer_{peer_id}_{peer_index}_model.pth')

        print(peer_models)
        FLPeer.aggregate(self, peer_models)