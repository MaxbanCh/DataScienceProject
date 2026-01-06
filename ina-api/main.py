import dataTreatment.INADataRecover as INADataRecover

def getDataForMonth(debut_date, end_date):
    data_recover = INADataRecover.INADataRecover()
    url = data_recover.getPersonforMonth(debut_date, end_date)
    return url

# Example usage
if __name__ == "__main__":
    # KEY = "0265604995556761"  # Replace with actual key
    
    # crypto = INACrypto(KEY)
    
    # # # Test encryption
    # filtres = {'channel': ['BFT', 'C+N', 'CIP', 'LCI', 'ITL'], 'minDate': '2024-07-01', 'maxDate': '2024-07-31', 'interval': 'day', 'top': 50, 'libpref': [], 'gender': ['Tous les genres', 'Hommes', 'Femmes']}

    
    # # encrypted = crypto.encrypt(filtres)
    # # print(f"Encrypted: {encrypted}")
    
    # # Test decryption

    # encrypted = "0329f4bb42ab291561391abfc6a4cf6ba359cea717444ef51824b7776e41fb0479b957c1fd4696c0bb1a5258d5b13bbf2667e23679013106ff1bc175aa8ee616ab737f4352fb1001ce73cce56cfac458a06f14470252f90728041825263befc59a77610dc7012c23378c14d41c7145af88c6df466e3a52597a584bab9497f0c7"
    
    # try:
    #     decrypted = crypto.decrypt(encrypted)
    #     print(f"Decrypted: {decrypted}")
    # except Exception as e:
    #     print(f"Decryption error: {e}")
    
    getDataForMonth('2024-06-01', '2024-07-31')

