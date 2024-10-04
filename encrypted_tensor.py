import torch
import tenseal as ts

class EncryptedTensor:
    def __init__(self, shape, encrypted_tensor_vector):
        self.shape = shape
        self.encrypted_tensor_vector = encrypted_tensor_vector
    
    def decrypt(self) -> torch.Tensor:
        encrypted_tensor_vector = self.encrypted_tensor_vector
        decrypted_tensor : ts.PlainTensor = encrypted_tensor_vector.decrypt()

        tensor : torch.Tensor = torch.tensor(decrypted_tensor)

        return tensor.reshape(self.shape)
    
    def encrypt(context, tensor: torch.Tensor):
        shape = tensor.shape
        tensor_vector = get_tensor_as_vector(tensor)
        
        return EncryptedTensor(shape, ts.ckks_vector(context, tensor_vector))
    
    def add(self, tensor: torch.Tensor):
        self.encrypted_tensor_vector = self.encrypted_tensor_vector.add(get_tensor_as_vector(tensor))
    
def get_tensor_as_vector(tensor: torch.Tensor):
    return tensor.reshape(1, get_shape_product(tensor.shape))[0]

def get_shape_product(shape):
    product = 1

    for i in range(0, len(shape), 1):
        product *= shape[i]
    
    return product