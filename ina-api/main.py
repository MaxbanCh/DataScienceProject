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

class INAAPIGenerator:
    def __init__(self, crypto):
        self.crypto = crypto
        self.base_url = "https://data-api.ina.fr/api/chart/"
    
    def getURLPersonforInfoChannel(self, debut_date, end_date):
        chartKey = "7d430f860c8f96f8ccecf84740815614"
        filters = {
            'channel': ['BFT', 'C+N', 'CIP', 'LCI', 'ITL'],
            'minDate': debut_date,
            'maxDate': end_date,
            'interval': 'day',
            'top': 50,
            'libpref': [],
            'gender': ['Tous les genres', 'Hommes', 'Femmes']
        }
        encrypted_filters = self.crypto.encrypt(filters)
        url = f"{self.base_url}{chartKey}?filters={encrypted_filters}"

        return url

    def getURLPersonforChannelJT(self, debut_date, end_date):
        chartKey = "af8dc595ac348f998be5fad12d829e98"
        filters = {
            'channel': ['ART', 'FR2', 'FR3', 'M6_', 'TF1'],
            'minDate': debut_date,
            'maxDate': end_date,
            'interval': 'day',
            'top': 50,
            'libpref': [],
            'gender': ['Tous les genres', 'Hommes', 'Femmes']
        }
        encrypted_filters = self.crypto.encrypt(filters)
        url = f"{self.base_url}{chartKey}?filters={encrypted_filters}"

        return url
    
    def getURLIterationWordforJT(self, debut_date, end_date, word):
        chartKey = "7d430f860c8f96f8ccecf84740815614"
        filters = {
            'minDate': debut_date,
            'maxDate': end_date,
            'interval': 'day',
            'libpref': [word],
        }
        encrypted_filters = self.crypto.encrypt(filters)
        url = f"{self.base_url}{chartKey}?filters={encrypted_filters}"

        return url
        
    def getURLIterationWordforContinue(self, debut_date, end_date, word):
        chartKey = "ea04b41e9da25fbb9090f7eea772d2b9"
        filters = {
            'minDate': debut_date,
            'maxDate': end_date,
            'interval': 'day',
            'libpref': [word],
        }
        encrypted_filters = self.crypto.encrypt(filters)
        url = f"{self.base_url}{chartKey}?filters={encrypted_filters}"

        return url

    def getURLProportionMenWomenChannelContinu(self, debut_date, end_date, channel):
        chartKey = "d8a4d8db9bef32cf105433de95fbf7e2"
        filters = {
            'channel': [channel], 
            'minDate': debut_date, 
            'maxDate': end_date, 
            'interval': 'day', 
            'media': 'chaines-information-continu'
        }
        encrypted_filters = self.crypto.encrypt(filters)
        url = f"{self.base_url}{chartKey}?filters={encrypted_filters}"

        return url
    
    def getURLProportionMenWomenChannelJT(self, debut_date, end_date, channel):
        chartKey = "d8a4d8db9bef32cf105433de95fbf7e2"
        filters = {
            'channel': [channel], 
            'minDate': debut_date, 
            'maxDate': end_date, 
            'interval': 'day', 
            'media': 'journaux-televises'
        }
        encrypted_filters = self.crypto.encrypt(filters)
        url = f"{self.base_url}{chartKey}?filters={encrypted_filters}"

        return url
    
    def getURLProportionMenWomenChannelRadio(self, debut_date, end_date, channel):
        chartKey = "d8a4d8db9bef32cf105433de95fbf7e2"
        filters = {
            'channel': [channel], 
            'minDate': debut_date, 
            'maxDate': end_date, 
            'interval': 'day', 
            'media': 'radios'
        }
        encrypted_filters = self.crypto.encrypt(filters)
        url = f"{self.base_url}{chartKey}?filters={encrypted_filters}"

        return url
    
class INAAPI:
    def __init__(self):
        self.key = "0265604995556761"
        self.crypto = INACrypto(self.key)
        self.apiGenerator = INAAPIGenerator(self.crypto)
        
        self.channelContinue = ['BFT', 'C+N', 'CIP', 'LCI', 'ITL']
        self.channelJT = ['ART', 'FR2', 'FR3', 'M6_', 'TF1']
        self.channelRadio = ['FCR', 'FIF', 'FIT', 'RMC', 'RTL']
    
    def getForMonth(self, debut_date, end_date):
        return self.apiGenerator.getURLPersonforInfoChannel(debut_date, end_date)
    


# Example usage
if __name__ == "__main__":
    KEY = "0265604995556761"  # Replace with actual key
    
    crypto = INACrypto(KEY)
    
    # # Test encryption
    filters = {'channel': ['BFT', 'C+N', 'CIP', 'LCI', 'ITL'], 'minDate': '2024-07-01', 'maxDate': '2024-07-31', 'interval': 'day', 'top': 50, 'libpref': [], 'gender': ['Tous les genres', 'Hommes', 'Femmes']}

    
    # encrypted = crypto.encrypt(filters)
    # print(f"Encrypted: {encrypted}")
    
    # Test decryption

    encrypted = "0329f4bb42ab291561391abfc6a4cf6ba359cea717444ef51824b7776e41fb0479b957c1fd4696c0bb1a5258d5b13bbf2667e23679013106ff1bc175aa8ee616ab737f4352fb1001ce73cce56cfac458a06f14470252f90728041825263befc59a77610dc7012c23378c14d41c7145af88c6df466e3a52597a584bab9497f0c7"
    
    try:
        decrypted = crypto.decrypt(encrypted)
        print(f"Decrypted: {decrypted}")
    except Exception as e:
        print(f"Decryption error: {e}")