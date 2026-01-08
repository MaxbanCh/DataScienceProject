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

    # Specific channel persons
    def getPersonforSpecificChannelJT(self, debut_date, end_date, channel):
        url = self.apiGenerator.getURLPersonforSpecificChannelJT(debut_date, end_date, channel)
        print(url)
        rawData = requests.get(self.apiGenerator.getURLPersonforSpecificChannelJT(debut_date, end_date, channel))
        try:
            dataByGroup = rawData.json()["data"][list(rawData.json()["data"].keys())[1]]
            dataByGroup = dataByGroup["data"]
            dataByGroup = dataByGroup["chartDatasRow"]
            newDataByGroup = []

            for point in dataByGroup:
                point['channel'] = channel
                newDataByGroup.append(point)
        
            return newDataByGroup

        except KeyError:
            print("No data available for the specified parameters.")
            return []
    
    def getPersonforSpecificChannelContinue(self, debut_date, end_date, channel):
        url = self.apiGenerator.getURLPersonforSpecificChannelContinue(debut_date, end_date, channel)
        print(url)
        rawData = requests.get(url)
        try:
            dataByGroup = rawData.json()["data"][list(rawData.json()["data"].keys())[1]]
            dataByGroup = dataByGroup["data"]
            dataByGroup = dataByGroup["chartDatasRow"]
            newDataByGroup = []

            for point in dataByGroup:
                point['channel'] = channel
                newDataByGroup.append(point)
        
            return newDataByGroup

        except KeyError:
            print("No data available for the specified parameters.")
            return []


    # Method for persons in group
    def getPersonforMonthJT(self, debut_date, end_date):
        allChannelData = []
        for channel in self.channelJT:
            channelData = self.getPersonforSpecificChannelJT(debut_date, end_date, channel)
            allChannelData.extend(channelData)
        print(allChannelData)
        return allChannelData
        # rawData = requests.get(self.apiGenerator.getURLPersonforChannelJT(debut_date, end_date))
        # dataByGroup = rawData.json()["data"][list(rawData.json()["data"].keys())[1]]["data"]["chartDatasRow"]
        # for point in dataByGroup:
        #     print(point)
        
        # return dataByGroup
    
    def getPersonforMonthContinue(self, debut_date, end_date):
        allChannelData = []
        for channel in self.channelContinue:
            channelData = self.getPersonforSpecificChannelContinue(debut_date, end_date, channel)
            allChannelData.extend(channelData)
        print(allChannelData)
        return allChannelData
    
        # rawData = requests.get(self.apiGenerator.getURLPersonforInfoChannel(debut_date, end_date))
        # dataByGroup = rawData.json()["data"][list(rawData.json()["data"].keys())[1]]["data"]["chartDatasRow"]
        # for point in dataByGroup:
        #     print(point)
        
        # return dataByGroup
    
    # Word Methods
    def getWordForMonthJT(self, debut_date, end_date, word):
        url = self.apiGenerator.getURLIterationWordforJT(debut_date, end_date, word)
        print(url)
        rawData = requests.get(url)
        dataByGroup = rawData.json()["data"][list(rawData.json()["data"].keys())[0]]["data"]["chartDatasRow"]
        print(dataByGroup)
        for point in dataByGroup:
            print(point)

        return dataByGroup

    def getWordForMonthContinue(self, debut_date, end_date, word):
        url = self.apiGenerator.getURLIterationWordforContinue(debut_date, end_date, word)
        print(url)
        rawData = requests.get(url)
        dataByGroup = rawData.json()["data"][list(rawData.json()["data"].keys())[0]]["data"]["chartDatasRow"]
        print(dataByGroup)
        for point in dataByGroup:
            print(point)

        return dataByGroup
    
    def getWomenMenProportionJT(self, debut_date, end_date, channel):
        url = self.apiGenerator.getURLProportionMenWomenChannelJT(debut_date, end_date, channel)
        print(url)
        rawData = requests.get(url)
        dataByGroup = rawData.json()["data"][list(rawData.json()["data"].keys())[1]]["data"]["chartDatasRow"]
        print(dataByGroup)
        for point in dataByGroup:
            print(point)

        return dataByGroup
    
    def getWomenMenProportionContinue(self, debut_date, end_date, channel):
        url = self.apiGenerator.getURLProportionMenWomenChannelContinu(debut_date, end_date, channel)
        print(url)
        rawData = requests.get(url)
        print(rawData.text)
        dataByGroup = rawData.json()["data"][list(rawData.json()["data"].keys())[1]]["data"]["chartDatasRow"]
        print(dataByGroup)
        for point in dataByGroup:
            print(point)

        return dataByGroup

    #### Fusion Methods for JT and continue
    def getPersonForMonth(self, debut_date, end_date):
        allData = []
        dataContinue = self.apiGenerator.getURLPersonforInfoChannel(debut_date, end_date)
        dataJT = self.apiGenerator.getURLPersonforChannelJT(debut_date, end_date)
        
        allData.extend(dataContinue)
        allData.extend(dataJT)
        return allData
    
    def getWordForMonth(self, debut_date, end_date, word):
        allData = []
        dataContinue = self.apiGenerator.getURLIterationWordforContinue(debut_date, end_date, word)
        dataJT = self.apiGenerator.getURLIterationWordforJT(debut_date, end_date, word)
        
        allData.extend(dataContinue)
        allData.extend(dataJT)
        return allData
