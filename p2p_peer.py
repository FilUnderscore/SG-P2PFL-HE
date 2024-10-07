from fl_peer import FLPeer
from MLTSModel import MLTSModel
from data_provider import CSVTSDataProvider

from threading import Thread
from flask import Flask, send_file, request

from requests import post, get
from shutil import copyfileobj

from time import sleep

import tenseal as ts
from io import BytesIO

from encrypted_model import EncryptedModel
from binary import BinaryDecoder, BinaryEncoder
from torch import div
from copy import deepcopy

app = Flask(__name__)
trained = False
peer_id = 0
peer_inst = None

# Aggregate model from another peer
@app.route('/aggregate_model', methods=['POST'])
def aggregate_model():
    buffer = BytesIO(request.data)
    decoder = BinaryDecoder(buffer)

    print('Received aggregate request.')

    model_owner = decoder.decode_int()
    aggregated_count = decoder.decode_int()
    encrypted_model = EncryptedModel.from_buffer(buffer, peer_inst.context)

    if peer_id != model_owner:
        peer_inst.aggregate_received_model(encrypted_model, model_owner, aggregated_count)
    else:
        peer_inst.decode_received_model(encrypted_model, aggregated_count)
    
    return "OK", 200

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
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40

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

    def fetch(self, url, target_filename):
        with get(url, stream=True) as request:
            with open(target_filename, 'wb') as file:
                copyfileobj(request.raw, file)

    def send(self, url, buffer):
        print('Posting to ' + url)
        post(url, data=buffer)

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
    
    def get_next_peer_id(self):
        peer_count = len(self.get_peer_list())
        return (peer_id + 1) % peer_count

    def aggregate_received_model(self, encrypted_model: EncryptedModel, model_owner: int, aggregated_count: int):
        state_dict = self.ml_model.model.model.state_dict()

        for k in encrypted_model.encrypted_tensors:
            encrypted_model.encrypted_tensors[k].add(state_dict[k])
        
        # send to next peer
        next_peer = self.get_peer_list()[self.get_next_peer_id()]

        encrypted_model_buffer = BytesIO()
        encoder = BinaryEncoder(encrypted_model_buffer)

        encoder.encode_int(model_owner)
        encoder.encode_int(aggregated_count + 1)
        
        self.send('http://' + next_peer['address'] + ':' + next_peer['ext_port'] + '/aggregate_model', encrypted_model.to_buffer(encrypted_model_buffer))

    def decode_received_model(self, encrypted_model: EncryptedModel, aggregated_count: int):
        state_dict = deepcopy(self.ml_model.model.model.state_dict())

        for k in encrypted_model.encrypted_tensors:
            encrypted_tensor = encrypted_model.encrypted_tensors[k]
            state_dict[k] = div(encrypted_tensor.decrypt(), aggregated_count + 1)
        
        self.ml_model.load_state_dict(state_dict)

    def aggregate(self):
        # send encrypted model to next peer and next peer will send to following peer etc.
        encrypted_model = self.ml_model.encrypt(self.context)

        next_peer = self.get_peer_list()[self.get_next_peer_id()]
        model_owner = peer_id

        print('Beginning encoding.')

        encrypted_model_buffer = BytesIO()
        encoder = BinaryEncoder(encrypted_model_buffer)

        encoder.encode_int(model_owner)
        encoder.encode_int(0)

        print('Beginning model aggregation.')

        self.send('http://' + next_peer['address'] + ':' + next_peer['ext_port'] + '/aggregate_model', encrypted_model.to_buffer(encrypted_model_buffer))

        print('Aggregated models.')