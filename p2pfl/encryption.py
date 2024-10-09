import torch
import tenseal as ts

from io import BytesIO
from p2pfl.binary import BinaryEncoder, BinaryDecoder

class EncryptedTensor:
    """Represents an encrypted torch tensor encrypted using HE (via tenseal)."""
    def __init__(self, shape, encrypted_tensor_vector):
        self.shape = shape
        self.encrypted_tensor_vector = encrypted_tensor_vector
    
    def decrypt(self) -> torch.Tensor:
        """Decrypts an encrypted torch tensor."""
        encrypted_tensor_vector = self.encrypted_tensor_vector
        decrypted_tensor : ts.PlainTensor = encrypted_tensor_vector.decrypt()

        tensor : torch.Tensor = torch.tensor(decrypted_tensor)

        return tensor.reshape(self.shape)
    
    def encrypt(context, tensor: torch.Tensor):
        """Encrypts a provided torch tensor."""
        shape = tensor.shape
        tensor_vector = get_tensor_as_vector(tensor)
        
        return EncryptedTensor(shape, ts.ckks_vector(context, tensor_vector))
    
    def add(self, tensor: torch.Tensor):
        """Adds an unencrypted torch tensor to the encrypted tensor."""
        self.encrypted_tensor_vector = self.encrypted_tensor_vector.add(get_tensor_as_vector(tensor))
    
    def encode(self, encoder: BinaryEncoder):
        """Encodes the encrypted tensor via the provided encoder."""
        shape = self.shape

        encoder.encode_int(len(shape))

        for i in range(0, len(shape), 1):
            encoder.encode_int(shape[i])
        
        encrypted_tensor_vector : ts.CKKSVector = self.encrypted_tensor_vector
        serialized_tensor_vector = encrypted_tensor_vector.serialize()

        encoder.encode_var_byte_array(serialized_tensor_vector)

    def decode(decoder: BinaryDecoder, context):
        """Decodes an encrypted tensor via the provided decoder."""
        tensor_shape_length = decoder.decode_int()
        tensor_shape = ()
        
        for i in range(0, tensor_shape_length, 1):
            tensor_shape = tensor_shape + (decoder.decode_int(),)
        
        serialized_tensor_vector = decoder.decode_var_byte_array()

        encrypted_tensor_vector : ts.CKKSVector = ts.ckks_vector_from(context, serialized_tensor_vector)
        return EncryptedTensor(tensor_shape, encrypted_tensor_vector)

def get_tensor_as_vector(tensor: torch.Tensor):
    """Converts a tensor into a vector for efficient HE computation."""
    return tensor.reshape(1, get_shape_product(tensor.shape))[0]

def get_shape_product(shape):
    """Calculates the product of all dimensions given a shape tuple."""
    product = 1

    for i in range(0, len(shape), 1):
        product *= shape[i]
    
    return product

class EncryptedModel:
    """Represents the encrypted state dict of a ML model."""
    def __init__(self, encrypted_tensors: dict[str, EncryptedTensor]):
        self.encrypted_tensors = encrypted_tensors

    def to_buffer(self, buffer = BytesIO()):
        """Encodes the encrypted model via the provided buffer."""
        encoder = BinaryEncoder(buffer)

        encrypted_tensors = self.encrypted_tensors

        encoder.encode_int(len(encrypted_tensors))

        for key in self.encrypted_tensors: # Encode each key-value pair in the model state dict
            encoder.encode_str(key)
            encrypted_tensors[key].encode(encoder)
        
        buffer.seek(0) # Reset the buffer position to 0 so that the buffer is read properly later on
        return buffer

    def from_buffer(buffer: BytesIO, context):
        """Decodes an encrypted model via the provided buffer."""
        decoder = BinaryDecoder(buffer)
        
        encrypted_tensors_length = decoder.decode_int()
        encrypted_tensors = {}

        for i in range(0, encrypted_tensors_length, 1): # Decode each key-value pair in the model state dict
            tensor_key = decoder.decode_str()
            encrypted_tensor = EncryptedTensor.decode(decoder, context)

            encrypted_tensors[tensor_key] = encrypted_tensor
        
        return EncryptedModel(encrypted_tensors)