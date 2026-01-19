import pandas as pd

file_path = 'List FWD.xlsx'
sheet_name = 'business_transactions'

try:
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=10)
    print("First 10 rows raw:")
    print(df.to_string())
    
    # We don't need the rest for now
    # print("Columns:", df.columns.tolist())
    # ...
except Exception as e:
    print(e)
