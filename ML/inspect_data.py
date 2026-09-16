import os
import pandas as pd
path='Data'
files=[f for f in os.listdir(path) if f.lower().endswith('.csv')]
print('files:', files)
for f in files:
    print('---', f)
    df=pd.read_csv(os.path.join(path,f), nrows=5)
    print(df.columns.tolist())
    print(df.head(2).to_string(index=False))
