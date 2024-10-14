# P2PFL
A novel Federated Learning framework for Python that leverages Peer-to-Peer networking via HTTP for privacy-preservation with Homomorphic Encryption.

## Getting Started
Run the `setup_dev.bat` script or run the following command:
```
pip install -r requirements.txt
pip install -e .
```

## How to use
Look at the examples provided in the `examples` folder which provide examples on using the framework both locally and using peer-to-peer networking with Homomorphic Encryption.

The model used in the examples is defined in the `examples/fl_sample_model.py`.

There are two examples included with the framework:
* `local_peers_example.py` involves setting up a variable number of local peers running on one machine.
* `p2p_example_peer.py` and `p2p_registration.py` involve setting up a peer-to-peer network with a central registration node utilizing Homomorphic Encryption (built into the p2p implementation) running on a variable number of machines in a network.

Both examples come with a number of scripts to run to demonstrate the framework in action.

## Credits
* Filip Jerkovic - author of P2PFL