import pandas as pd

#Read the file
df = pd.read_excel('AB_NYC_2019.xlsx')

#Dirty data count output
print(df.isna().sum())
print("Total dirty data count",sum(df.isna().sum()))



