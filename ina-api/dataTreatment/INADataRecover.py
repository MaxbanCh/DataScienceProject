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
        print("Person", channel, url)
        rawData = requests.get(self.apiGenerator.getURLPersonforSpecificChannelJT(debut_date, end_date, channel))
        if rawData.status_code != 200:
            print(f"API returned status code {rawData.status_code} for {channel}")
            return []
        
        if not rawData.text:
            print(f"Empty response for {channel}")
            return []


        try:
            dataByGroup = rawData.json()
            dataByGroup = dataByGroup["data"][list(dataByGroup["data"].keys())[1]]
            dataByGroup = dataByGroup["data"]
            dataByGroup = dataByGroup["chartDatasRow"]
            newDataByGroup = []

            for point in dataByGroup:
                if point['value'] != 0:
                    point['channel'] = channel
                    newDataByGroup.append(point)
        
            return newDataByGroup

        except KeyError:
            print("No data available for the specified parameters.", channel)
            return []
    

    def getPersonforSpecificChannelContinue(self, debut_date, end_date, channel):
        url = self.apiGenerator.getURLPersonforSpecificChannelContinue(debut_date, end_date, channel)
        print("Person", channel, url)
        rawData = requests.get(url)
        if rawData.status_code != 200:
            print(f"API returned status code {rawData.status_code} for {channel}")
            return []
        
        if not rawData.text:
            print(f"Empty response for {channel}")
            return []

        try:
            dataByGroup = rawData.json()
            dataByGroup = dataByGroup["data"][list(dataByGroup["data"].keys())[1]]
            dataByGroup = dataByGroup["data"]
            dataByGroup = dataByGroup["chartDatasRow"]
            newDataByGroup = []

            for point in dataByGroup:
                if point['value'] != 0:
                    point['channel'] = channel
                    newDataByGroup.append(point)
        
            return newDataByGroup

        except KeyError:
            print("No data available for the specified parameters.", channel)
            return []

    def getWomenMenProportionSpecificChannelJT(self, debut_date, end_date, channel):
        url = self.apiGenerator.getURLProportionMenWomenChannelJT(debut_date, end_date, channel)
        print("Women Prop", channel, url)
        rawData = requests.get(url)
        if rawData.status_code != 200:
            print(f"API returned status code {rawData.status_code} for {channel}")
            return []
        
        if not rawData.text:
            print(f"Empty response for {channel}")
            return []
        
        try:
            dataByGroup = rawData.json()
            dataByGroup = dataByGroup["data"][list(dataByGroup["data"].keys())[1]]["data"]["chartDatasRow"]
            newDataByGroup = []
            for point in dataByGroup:
                point['channel'] = channel
                newDataByGroup.append(point)

            return newDataByGroup
        except KeyError:
            print("No data available for the specified parameters.", channel)
            return []
    
    def getWomenMenProportionSpecificContinue(self, debut_date, end_date, channel):
        url = self.apiGenerator.getURLProportionMenWomenChannelContinu(debut_date, end_date, channel)
        print("Women Prop", channel, url)
        rawData = requests.get(url)

        if rawData.status_code != 200:
            print(f"API returned status code {rawData.status_code} for {channel}")
            return []
        
        if not rawData.text:
            print(f"Empty response for {channel}")
            return []
        
        try:
            dataByGroup = rawData.json()
            dataByGroup = dataByGroup["data"][list(dataByGroup["data"].keys())[1]]["data"]["chartDatasRow"]
            newDataByGroup = []
            for point in dataByGroup:
                point['channel'] = channel
                newDataByGroup.append(point)

            return newDataByGroup
        except KeyError:
            print("No data available for the specified parameters.", channel)
            return []

    # Method for persons in group
    def getPersonforMonthJT(self, debut_date, end_date):
        allChannelData = []
        for channel in self.channelJT:
            channelData = self.getPersonforSpecificChannelJT(debut_date, end_date, channel)
            allChannelData.extend(channelData)
        # print(allChannelData)
        return allChannelData
    
    def getPersonforMonthContinue(self, debut_date, end_date):
        allChannelData = []
        for channel in self.channelContinue:
            channelData = self.getPersonforSpecificChannelContinue(debut_date, end_date, channel)
            allChannelData.extend(channelData)
        # print(allChannelData)
        return allChannelData
    
    def getWomenMenProportionJT(self, debut_date, end_date):
        allChannelData = []
        for channel in self.channelJT:
            channelData = self.getWomenMenProportionSpecificChannelJT(debut_date, end_date, channel)
            allChannelData.extend(channelData)
        # print(allChannelData)
        return allChannelData
    
    def getWomenMenProportionContinue(self, debut_date, end_date):
        allChannelData = []
        for channel in self.channelContinue:
            channelData = self.getWomenMenProportionSpecificContinue(debut_date, end_date, channel)
            allChannelData.extend(channelData)
        # print(allChannelData)
        return allChannelData
    
    # Word Methods
    def getWordForMonthJT(self, debut_date, end_date, word):
        url = self.apiGenerator.getURLIterationWordforJT(debut_date, end_date, word)
        print("Word", word, url)
        rawData = requests.get(url)
        if rawData.status_code != 200:
            print(f"API returned status code {rawData.status_code} for {word}")
            return []
        
        if not rawData.text:
            print(f"Empty response for {word}")
            return []
        
        try:
            dataByGroup = rawData.json()
            dataByGroup = dataByGroup["data"][list(dataByGroup["data"].keys())[0]]["data"]["chartDatasRow"]
            # print(dataByGroup)

            return dataByGroup
        except KeyError:
            print("No data available for the specified parameters.", word)

    def getWordForMonthContinue(self, debut_date, end_date, word):
        url = self.apiGenerator.getURLIterationWordforContinue(debut_date, end_date, word)
        print("Word", word, url)
        rawData = requests.get(url)
        if rawData.status_code != 200:
            print(f"API returned status code {rawData.status_code} for {word}")
            return []
        
        if not rawData.text:
            print(f"Empty response for {word}")
            return []
        
        try:
            dataByGroup = rawData.json()
            dataByGroup = dataByGroup["data"][list(dataByGroup["data"].keys())[0]]["data"]["chartDatasRow"]
            # print(dataByGroup)

            return dataByGroup
        except KeyError:
            print("No data available for the specified parameters.", word)
            return []
        

    #### Fusion Methods for JT and continue
    def getPersonForMonth(self, debut_date, end_date):
        allData = []
        dataContinue = self.getPersonforMonthContinue(debut_date, end_date)
        dataJT = self.getPersonforMonthJT(debut_date, end_date)
        
        allData.extend(dataContinue)
        allData.extend(dataJT)
        return allData
    
    def getWordForMonth(self, debut_date, end_date, word):
        allData = []
        dataContinue = self.getWordForMonthContinue(debut_date, end_date, word)
        dataJT = self.getWordForMonthJT(debut_date, end_date, word)
        
        allData.extend(dataContinue)
        allData.extend(dataJT)
        return allData

    def getWomenMenProportion(self, debut_date, end_date):
        allData = []
        dataContinue = self.getWomenMenProportionContinue(debut_date, end_date)
        dataJT = self.getWomenMenProportionJT(debut_date, end_date)
        
        allData.extend(dataContinue)
        allData.extend(dataJT)
        return allData