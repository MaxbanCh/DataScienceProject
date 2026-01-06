import INACrypto
import json

class INAAPIGenerator:
    def __init__(self, crypto : INACrypto.INACrypto):
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
