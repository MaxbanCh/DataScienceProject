from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import json

class INACrypto:
    def __init__(self, key):
        """
        key: The encryption key string (find it in ENCRYPTED_KEY)
        """
        self.key = key.encode('utf-8')
        
    def encrypt(self, data):
        """
        Encrypt data (dict or string) to hex format for INA API
        """
        # Convert dict to JSON string if needed
        if isinstance(data, dict):
            data = json.dumps(data, separators=(',', ':'))
        
        # Create cipher in ECB mode
        cipher = AES.new(self.key, AES.MODE_ECB)
        
        # Pad and encrypt
        padded = pad(data.encode('utf-8'), AES.block_size)
        encrypted = cipher.encrypt(padded)
        
        # Return as hex string
        return encrypted.hex()
    
    def decrypt(self, hex_string):
        """
        Decrypt hex string from INA API response
        """
        # Convert hex to bytes
        ciphertext = bytes.fromhex(hex_string)
        
        # Create cipher in ECB mode
        cipher = AES.new(self.key, AES.MODE_ECB)
        
        # Decrypt and unpad
        decrypted = cipher.decrypt(ciphertext)
        unpadded = unpad(decrypted, AES.block_size)
        
        # Return as string (try to parse as JSON)
        result = unpadded.decode('utf-8')
        try:
            return json.loads(result)
        except:
            return result
        