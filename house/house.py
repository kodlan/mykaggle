import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2500)

train = pd.read_csv('.\\data\\train.csv')
test = pd.read_csv('.\\data\\test.csv')

print(train.iloc[:10, :200])

