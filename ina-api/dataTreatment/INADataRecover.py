from API.INACrypto import INACrypto
from API.INAAPIGenerator import INAAPIGenerator
import requests

# Date format: 'YYYY-MM-DD'

class INADataRecover:
    def __init__(self):
        self.key = "0265604995556761"
        self.crypto = INACrypto(self.key)
        self.apiGenerator = INAAPIGenerator(self.crypto)
        
        self.channelContinue = ['BFT', 'C+N', 'CIP', 'LCI', 'ITL']
        self.channelJT = ['ART', 'FR2', 'FR3', 'M6_', 'TF1']
        self.channelRadio = ['FCR', 'FIF', 'FIT', 'RMC', 'RTL']

    def getPersonforMonth(self, debut_date, end_date):
        rawData = requests.get(self.apiGenerator.getURLPersonforInfoChannel(debut_date, end_date))
        dataByGroup = rawData.json()["data"][list(rawData.json()["data"].keys())[1]]["data"]["chartDatasRow"]
        for point in dataByGroup:
            print(point)
        
        return dataByGroup

            
    
    def getForMonth(self, debut_date, end_date):
        return self.apiGenerator.getURLPersonforInfoChannel(debut_date, end_date)