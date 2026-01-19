import pandas as pd

# Read the Excel file
df = pd.read_excel(r'd:\CODE\pkl\List FWD.xlsx')

# Display basic information
print('Total rows:', len(df))
print('Total columns:', len(df.columns))
print('\nColumn names:')
for i, col in enumerate(df.columns, 1):
    print(f'{i}. {col}')

print('\n' + '='*100)
print('DATA PREVIEW:')
print('='*100)

# Display data in a more readable format
for idx, row in df.iterrows():
    print(f"\n--- Row {idx} ---")
    for col in df.columns:
        value = row[col]
        if pd.notna(value):
            print(f"  {col}: {value}")
