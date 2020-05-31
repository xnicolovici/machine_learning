import pandas as pd


df = pd.read_csv('/home/xavier/Documents/Formations/Machine Learning/data/titanic.csv', index_col='PassengerId')



print(df.head(3))
