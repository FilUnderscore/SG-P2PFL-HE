import tenseal as ts
from io import BytesIO
import numpy as np
from binary import BinaryEncoder, BinaryDecoder
from encrypted_tensor import EncryptedTensor

class EncryptedModel:
    def __init__(self, encrypted_tensors: dict[str, EncryptedTensor]):
        self.encrypted_tensors = encrypted_tensors

    def to_buffer(self, buffer = BytesIO()):
        encoder = BinaryEncoder(buffer)

        encrypted_tensors = self.encrypted_tensors

        encoder.encode_int(len(encrypted_tensors))

        for key in self.encrypted_tensors:
            encoder.encode_tensor(key, encrypted_tensors[key])

        buffer.seek(0)
        return buffer

    def from_buffer(buffer: BytesIO, context):
        decoder = BinaryDecoder(buffer)
        
        encrypted_tensors_length = decoder.decode_int()
        encrypted_tensors = {}

        for i in range(0, encrypted_tensors_length, 1):
            encrypted_tensor = decoder.decode_tensor(context)
            encrypted_tensors[encrypted_tensor[0]] = encrypted_tensor[1]
        
        return EncryptedModel(encrypted_tensors)