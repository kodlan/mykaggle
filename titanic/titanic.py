import pandas as pd
import keras as K
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler


class MyLogger(K.callbacks.Callback):
  def __init__(self, n):
    self.n = n   # print loss & acc every n epochs

  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.n == 0:
      curr_loss =logs.get('loss')
      curr_acc = logs.get('acc') * 100
      print("epoch = %4d loss = %0.6f acc = %0.2f%%" % (epoch, curr_loss, curr_acc))


def categorize(data, column):
    data[column] = pd.Categorical(data[column])
    dfDummies = pd.get_dummies(data[column], prefix=column)
    return pd.concat([data, dfDummies], axis=1)


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1500)

data = pd.read_csv(".\\titanic\\data\\train.csv")

# Fill all NaNs with 0
data['Cabin'] = data['Cabin'].fillna('-')
data['Age'] = data['Age'].fillna(data['Age'].mean())
data = data.fillna(0)


# Categorize
data = categorize(data, 'Pclass')
data = categorize(data, 'Sex')
data = categorize(data, 'Embarked')
data['Cabin'] = data['Cabin'].apply((lambda x: x[0] if len(x) > 0 else '0'))
data = categorize(data, 'Cabin')

print(data.iloc[:10, :200])

# Create np arrays
X = data.as_matrix(columns=['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'Age', 'SibSp', 'Parch', 'Fare',
                            'Embarked_0', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Cabin_-', 'Cabin_A', 'Cabin_B',
                            'Cabin_C', 'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T'])
Y = data.as_matrix(columns=['Survived'])

# Normalize
sc = StandardScaler()
X = sc.fit_transform(X)

print(X.shape)
print(Y.shape)

# Prepare model
model = Sequential()
model.add(Dense(4, input_dim=22, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=K.optimizers.SGD(lr=0.01), metrics=['accuracy'])

my_logger = MyLogger(n=100)
h = model.fit(X, Y, batch_size=10, epochs=1000, verbose=0, callbacks=[my_logger])

test = pd.read_csv(".\\titanic\\data\\test.csv")

# Fill all NaNs with 0
test['Cabin'] = test['Cabin'].fillna('-')
test = test.fillna(0)

# Categorize
test = categorize(test, 'Pclass')
test = categorize(test, 'Sex')
test = categorize(test, 'Embarked')
test['Cabin'] = test['Cabin'].apply((lambda x: x[0] if len(x) > 0 else '0'))
test = categorize(test, 'Cabin')


test['Cabin_T'] = 0
test['Embarked_0'] = 0

print(test.iloc[:10, :200])

# Create np arrays
T = test.as_matrix(columns=['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'Age', 'SibSp', 'Parch', 'Fare',
                            'Embarked_0', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Cabin_-', 'Cabin_A', 'Cabin_B',
                            'Cabin_C', 'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T'])
# Normalize
T = sc.fit_transform(T)

pred = model.predict(T)
test['Survived'] = np.round(pred).astype(int)

test.to_csv('.\\titanic\\output.csv', columns=['PassengerId', 'Survived'], index=False)
