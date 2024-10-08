from io import BytesIO

class BinaryEncoder:
    def __init__(self, buffer: BytesIO):
        self.buffer = buffer

    def encode_int(self, value: int):
        self.buffer.write(value.to_bytes(4))
    
    def encode_var_byte_array(self, value: bytes):
        self.encode_int(len(value))
        self.buffer.write(value)
    
    def encode_str(self, value: str):
        self.encode_var_byte_array(value.encode())

class BinaryDecoder:
    def __init__(self, buffer: BytesIO):
        self.buffer = buffer
    
    def decode_int(self) -> int:
        return int.from_bytes(self.buffer.read(4))
    
    def decode_var_byte_array(self) -> bytes:
        return self.buffer.read(self.decode_int())
    
    def decode_str(self) -> str:
        return self.decode_var_byte_array().decode()