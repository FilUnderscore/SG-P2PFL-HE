from io import BytesIO

class BinaryEncoder:
    """Represents a buffer which can encode data as bytes."""
    def __init__(self, buffer: BytesIO):
        self.buffer = buffer

    def encode_int(self, value: int):
        """Encodes an int value into 4 bytes, into the buffer."""
        self.buffer.write(value.to_bytes(4))
    
    def encode_var_byte_array(self, value: bytes):
        """Encodes a variable byte array given a length, into the buffer."""
        self.encode_int(len(value))
        self.buffer.write(value)
    
    def encode_str(self, value: str):
        """Encodes a UTF-8 string into a variable byte array."""
        self.encode_var_byte_array(value.encode())

class BinaryDecoder:
    """Represents a buffer which can decode data from bytes."""
    def __init__(self, buffer: BytesIO):
        self.buffer = buffer
    
    def decode_int(self) -> int:
        """Decodes an int value from the next 4 bytes of the buffer."""
        return int.from_bytes(self.buffer.read(4))
    
    def decode_var_byte_array(self) -> bytes:
        """Decodes a variable byte array given a decoded length from the buffer."""
        return self.buffer.read(self.decode_int())
    
    def decode_str(self) -> str:
        """Decodes a UTF-8 string from a variable byte array."""
        return self.decode_var_byte_array().decode()