from io import BytesIO
from encrypted_tensor import EncryptedTensor
import tenseal as ts

class BinaryEncoder:
    def __init__(self, buffer: BytesIO):
        self.buffer = buffer

    def encode_int(self, value: int):
        self.buffer.write(value.to_bytes(4))
    
    def encode_str(self, value: str):
        encoded_str = value.encode()
        self.encode_int(len(encoded_str))
        self.buffer.write(encoded_str)
    
    def encode_tensor(self, key: str, tensor: EncryptedTensor):
        self.encode_str(key)

        shape = tensor.shape

        self.encode_int(len(shape))

        for i in range(0, len(shape), 1):
            self.encode_int(shape[i])
        
        encrypted_tensor_vector : ts.CKKSVector = tensor.encrypted_tensor_vector
        serialized_tensor_vector = encrypted_tensor_vector.serialize()

        self.encode_int(len(serialized_tensor_vector))
        self.buffer.write(serialized_tensor_vector)

class BinaryDecoder:
    def __init__(self, buffer: BytesIO):
        self.buffer = buffer
    
    def decode_int(self) -> int:
        return int.from_bytes(self.buffer.read(4))
    
    def decode_str(self) -> str:
        return self.buffer.read(self.decode_int()).decode()
    
    def decode_tensor(self, context) -> tuple[str, EncryptedTensor]:
        tensor_key = self.decode_str()

        tensor_shape_length = self.decode_int()
        tensor_shape = ()

        for i in range(0, tensor_shape_length, 1):
            tensor_shape = tensor_shape + (self.decode_int(),)
        
        serialized_tensor_vector_length = self.decode_int()
        serialized_tensor_vector = self.buffer.read(serialized_tensor_vector_length)

        encrypted_tensor_vector : ts.CKKSVector = ts.ckks_vector_from(context, serialized_tensor_vector)
        return (tensor_key, EncryptedTensor(tensor_shape, encrypted_tensor_vector))