import dataTreatment.INADataRecover as INADataRecover
import pandas as pd
import os

class DataSave:
    def __init__(self):
        self.dataRecover = INADataRecover.INADataRecover()
    
    def createDirIfNotExists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    def convertJSONtoCSV(self, data):
        df = pd.DataFrame(data)
        # print(df.head())
        return df

    def saveDataPersonChannels(self, debut_date, end_date):
        data = self.dataRecover.getPersonForMonth(debut_date, end_date)
        df = self.convertJSONtoCSV(data)
        self.createDirIfNotExists("data/INA/People")
        df.to_csv(f"data/INA/People/person_most_quoted_{debut_date}_to_{end_date}.csv", index=False)

    def saveDataWordChannels(self, debut_date, end_date, word):
        data = self.dataRecover.getWordForMonth(debut_date, end_date, word)
        df = self.convertJSONtoCSV(data)
        self.createDirIfNotExists("data/INA/Words")
        df.to_csv(f"data/INA/Words/word_{word}_channel_JT_{debut_date}_to_{end_date}.csv", index=False)

    def saveWomenMenProportion(self, debut_date, end_date):
        data = self.dataRecover.getWomenMenProportion(debut_date, end_date)
        df = self.convertJSONtoCSV(data)
        self.createDirIfNotExists("data/INA/WomenMenProportion")
        df.to_csv(f"data/INA/WomenMenProportion/women_men_proportion_{debut_date}_to_{end_date}.csv", index=False)