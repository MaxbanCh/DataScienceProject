from dataTreatment.AutoWork import AutoWork

# Exeample of a possible exectution
if __name__ == "__main__":
    autoWork = AutoWork(begin_year=2024, end_year=2024, begin_month=5, end_month=12)
    autoWork.savePersonChannelData()
    # autoWork.saveWomenMenProportionData()
    # autoWork.getWordChannelData('Election')