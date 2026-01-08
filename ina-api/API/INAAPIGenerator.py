import API.INACrypto as INACrypto
import json

class INAAPIGenerator:
    def __init__(self, crypto : INACrypto.INACrypto):
        self.crypto = crypto
        self.base_url = "https://data-api.ina.fr/api/chart/"
    
    def getURLPersonforInfoChannel(self, debut_date, end_date):
        chartKey = "7d430f860c8f96f8ccecf84740815614"
        filtres = {
            'channel': ['BFT', 'C+N', 'CIP', 'LCI', 'ITL'],
            'minDate': debut_date,
            'maxDate': end_date,
            'interval': 'day',
            'top': 50,
            'libpref': [],
            'gender': ['Tous les genres', 'Hommes', 'Femmes']
        }
        encrypted_filtres = self.crypto.encrypt(filtres)
        url = f"{self.base_url}{chartKey}?filtres={encrypted_filtres}"

        return url

    def getURLPersonforChannelJT(self, debut_date, end_date):
        chartKey = "af8dc595ac348f998be5fad12d829e98"
        filtres = {
            'channel': ['ART', 'FR2', 'FR3', 'M6_', 'TF1'],
            'minDate': debut_date,
            'maxDate': end_date,
            'interval': 'day',
            'top': 50,
            'libpref': [],
            'gender': ['Tous les genres', 'Hommes', 'Femmes']
        }
        encrypted_filtres = self.crypto.encrypt(filtres)
        url = f"{self.base_url}{chartKey}?filtres={encrypted_filtres}"

        return url

    def getURLPersonforSpecificChannelJT(self, debut_date, end_date, channel):
        chartKey = "af8dc595ac348f998be5fad12d829e98"
        filtres = {
            'channel': [channel],
            'minDate': debut_date,
            'maxDate': end_date,
            'interval': 'day',
            'top': 50,
            'libpref': [],
            'gender': ['Tous les genres', 'Hommes', 'Femmes']
        }
        encrypted_filtres = self.crypto.encrypt(filtres)
        url = f"{self.base_url}{chartKey}?filtres={encrypted_filtres}"

        return url
    
    def getURLPersonforSpecificChannelContinue(self, debut_date, end_date, channel):
        chartKey = "7d430f860c8f96f8ccecf84740815614"
        filtres = {
            'channel': [channel],
            'minDate': debut_date,
            'maxDate': end_date,
            'interval': 'day',
            'top': 50,
            'libpref': [],
            'gender': ['Tous les genres', 'Hommes', 'Femmes']
        }
        encrypted_filtres = self.crypto.encrypt(filtres)
        url = f"{self.base_url}{chartKey}?filtres={encrypted_filtres}"

        return url
    
    def getURLPersonforSpecificChannelContinue(self, debut_date, end_date, channel):
        chartKey = "af8dc595ac348f998be5fad12d829e98"
        filtres = {
            'channel': [channel],
            'minDate': debut_date,
            'maxDate': end_date,
            'interval': 'day',
            'top': 50,
            'libpref': [],
            'gender': ['Tous les genres', 'Hommes', 'Femmes']
        }
        encrypted_filtres = self.crypto.encrypt(filtres)
        url = f"{self.base_url}{chartKey}?filtres={encrypted_filtres}"

        return url
    
    def getURLIterationWordforJT(self, debut_date, end_date, word):
        chartKey = "36285fc811c7fba5cf14854cc38e8188"
        filtres = {
            'minDate': debut_date,
            'maxDate': end_date,
            'interval': 'day',
            'libpref': [word],
        }
        encrypted_filtres = self.crypto.encrypt(filtres)
        url = f"{self.base_url}{chartKey}?filtres={encrypted_filtres}"

        return url
        
    def getURLIterationWordforContinue(self, debut_date, end_date, word):
        chartKey = "ea04b41e9da25fbb9090f7eea772d2b9"
        filtres = {
            'minDate': debut_date,
            'maxDate': end_date,
            'interval': 'day',
            'libpref': [word],
        }
        encrypted_filtres = self.crypto.encrypt(filtres)
        url = f"{self.base_url}{chartKey}?filtres={encrypted_filtres}"

        return url

    def getURLProportionMenWomenChannelContinu(self, debut_date, end_date, channel):
        chartKey = "f579cec8bf6bfa001ec09f04dbbb5907"
        filtres = {
            'channel': [channel], 
            'minDate': debut_date, 
            'maxDate': end_date, 
            'interval': 'day', 
            'media': 'chaines-information-continu'
        }
        encrypted_filtres = self.crypto.encrypt(filtres)
        url = f"{self.base_url}{chartKey}?filtres={encrypted_filtres}"

        return url
    
    def getURLProportionMenWomenChannelJT(self, debut_date, end_date, channel):
        chartKey = "f579cec8bf6bfa001ec09f04dbbb5907"
        filtres = {
            'channel': [channel], 
            'minDate': debut_date, 
            'maxDate': end_date, 
            'interval': 'day', 
            'media': 'journaux-televises'
        }
        encrypted_filtres = self.crypto.encrypt(filtres)
        url = f"{self.base_url}{chartKey}?filtres={encrypted_filtres}"

        return url
    
    def getURLProportionMenWomenChannelRadio(self, debut_date, end_date, channel):
        chartKey = "d8a4d8db9bef32cf105433de95fbf7e2"
        filtres = {
            'channel': [channel], 
            'minDate': debut_date, 
            'maxDate': end_date, 
            'interval': 'day', 
            'media': 'radios'
        }
        encrypted_filtres = self.crypto.encrypt(filtres)
        url = f"{self.base_url}{chartKey}?filtres={encrypted_filtres}"

        return url
