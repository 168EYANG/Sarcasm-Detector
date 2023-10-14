import pandas as pd

for chunk in pd.read_csv("sarc.csv", on_bad_lines='skip', chunksize=5):
    print(chunk)
    break