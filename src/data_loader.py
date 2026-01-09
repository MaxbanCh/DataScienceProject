import pandas as pd
import os

"""
Utility class to inspect Excel files, iterates through sheets and extract them.
"""

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.excel_file = pd.ExcelFile(file_path)
    """    
    Return the name of all sheets in Excel file
    """
    def list_sheets(self):
        return list(self.excel_file.sheet_names)

    """
    Extract a specific sheet from Excel file into a dataframe
    """
    def extract_sheets(self, sheet_name, skip_rows=0):
        df = pd.read_excel(self.excel_file, sheet_name=sheet_name, skiprows=skip_rows)
        # Normalize column names
        df.columns = [str(c).strip().replace(' ', '_').lower() for c in df.columns]
        return df