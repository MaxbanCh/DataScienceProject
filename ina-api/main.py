import dataTreatment.INADataRecover as INADataRecover
import API.INACrypto as INACrypto

def getDataForMonth(debut_date, end_date, word):
    data_recover = INADataRecover.INADataRecover()
    url = data_recover.getWomenMenProportionContinue(debut_date, end_date, word)
    return url


if __name__ == "__main__":
    # KEY = "0265604995556761"  # Replace with actual key
    
    # crypto = INACrypto.INACrypto(KEY)
    
    # # # # Test encryption
    # # filtres = {'channel': ['BFT', 'C+N', 'CIP', 'LCI', 'ITL'], 'minDate': '2024-07-01', 'maxDate': '2024-07-31', 'interval': 'day', 'top': 50, 'libpref': [], 'gender': ['Tous les genres', 'Hommes', 'Femmes']}

    
    # # # encrypted = crypto.encrypt(filtres)
    # # # print(f"Encrypted: {encrypted}")
    
    # # # Test decryption

    # encrypted = "2e4748bc81b59c7227c83050bcce95d879b0aec6d04dd8afffe67ad5d32c78ad92bd15c9627a067ce83df045136e4274d9f49eaab767734aeab5682daf4bebbaab737f4352fb1001ce73cce56cfac458a06f14470252f90728041825263befc5a91cb5161b6cf5c851a686539e52403dfd7f2b1ef4652a6cadee5d0f7e1886f875189095c34f2ab3046459c676c27395"
    
    # try:
    #     decrypted = crypto.decrypt(encrypted)
    #     print(f"Decrypted: {decrypted}")
    # except Exception as e:
    #     print(f"Decryption error: {e}")
    
    getDataForMonth('2024-06-01', '2024-07-31', 'C+N')

