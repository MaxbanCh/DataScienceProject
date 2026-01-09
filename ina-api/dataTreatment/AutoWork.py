import dataTreatment.DataSave as DataSave

class AutoWork:
    def __init__(self, begin_year, end_year):
        self.begin_year = begin_year
        self.end_year = end_year

        self.months = [
            "01", "02", "03", "04", "05", "06",
            "07", "08", "09", "10", "11", "12"
        ]
        self.days_in_month = {
            "01": 31, "02": 28, "03": 31, "04": 30,
            "05": 31, "06": 30, "07": 31, "08": 31,
            "09": 30, "10": 31, "11": 30, "12": 31
        }

        self.dataSave = DataSave.DataSave()

    def savePersonChannelData(self):
        for year in range(self.begin_year, self.end_year + 1):
            for month in self.months:
                debut_date = f"{year}-{month}-01"
                end_day = self.days_in_month[month]
                end_date = f"{year}-{month}-{end_day}"
                self.dataSave.saveDataPersonChannels(debut_date, end_date)
    
    def saveWomenMenProportionData(self):
        for year in range(self.begin_year, self.end_year + 1):
            for month in self.months:
                debut_date = f"{year}-{month}-01"
                end_day = self.days_in_month[month]
                end_date = f"{year}-{month}-{end_day}"
                self.dataSave.saveWomenMenProportion(debut_date, end_date)

    def getWordChannelData(self, word):
        for year in range(self.begin_year, self.end_year + 1):
            for month in self.months:
                debut_date = f"{year}-{month}-01"
                end_day = self.days_in_month[month]
                end_date = f"{year}-{month}-{end_day}"
                self.dataSave.saveDataWordChannels(debut_date, end_date, word)