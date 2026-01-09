from multiprocessing import Pool

import dataTreatment.DataSave as DataSave

class AutoWork:
    def __init__(self, begin_year, end_year, begin_month=1, end_month=12):
        self.begin_year = begin_year
        self.end_year = end_year
        self.begin_month = begin_month
        self.end_month = end_month

        self.months = [
            f"{i:02d}" for i in range(self.begin_month, self.end_month + 1)
        ]
        self.days_in_month = {
            "01": 31, "02": 28, "03": 31, "04": 30,
            "05": 31, "06": 30, "07": 31, "08": 31,
            "09": 30, "10": 31, "11": 30, "12": 31
        }

        self.dataSave = DataSave.DataSave()
    
    def process_person_month(self, args):
        year, month = args
        debut_date = f"{year}-{month}-01"
        end_day = self.days_in_month[month]
        end_date = f"{year}-{month}-{end_day}"
        self.dataSave.saveDataPersonChannels(debut_date, end_date)    

    def process_proportion_month(self, args):
        year, month = args
        debut_date = f"{year}-{month}-01"
        end_day = self.days_in_month[month]
        end_date = f"{year}-{month}-{end_day}"
        self.dataSave.saveWomenMenProportion(debut_date, end_date)

    def savePersonChannelData(self):
        tasks = [(year, month) for year in range(self.begin_year, self.end_year + 1) for month in self.months]
        
        with Pool() as pool:
            pool.map(self.process_person_month, tasks)

    def saveWomenMenProportionData(self):
        tasks = [(year, month) for year in range(self.begin_year, self.end_year + 1) for month in self.months]
        
        with Pool() as pool:
            pool.map(self.process_proportion_month, tasks)

    def getWordChannelData(self, word):
        for year in range(self.begin_year, self.end_year + 1):
            for month in self.months:
                debut_date = f"{year}-{month}-01"
                end_day = self.days_in_month[month]
                end_date = f"{year}-{month}-{end_day}"
                self.dataSave.saveDataWordChannels(debut_date, end_date, word)