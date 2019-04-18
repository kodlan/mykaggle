import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2500)

train = pd.read_csv('.\\data\\train.csv')
test = pd.read_csv('.\\data\\test.csv')

print(train.iloc[:10, :200])
print(train.SalePrice.describe())

nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

categoricals = train.select_dtypes(exclude=[np.number])
categoricals.describe()

train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

print ('Encoded: \n')
print (train.enc_street.value_counts())


def encode(x): return 1 if x == 'Partial' else 0


train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)

data = train.select_dtypes(include=[np.number]).interpolate().dropna()

y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

lr = linear_model.LinearRegression()

model = lr.fit(X_train, y_train)

predictions = model.predict(X_test)

print ('RMSE is: \n', mean_squared_error(y_test, predictions))

submission = pd.DataFrame()
submission['Id'] = test.Id

feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()

predictions = model.predict(feats)
final_predictions = np.exp(predictions)

print ("Original predictions are: \n", predictions[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])

submission['SalePrice'] = final_predictions
submission.to_csv('result.csv', index=False)