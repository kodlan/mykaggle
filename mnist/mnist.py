import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2500)

train = pd.read_csv('.\\data\\train.csv')
test = pd.read_csv('.\\data\\test.csv')

print(train.iloc[:10, :200])

#todo: use keras instead
sc = StandardScaler()

X_train = train.values[0:40000, 1:]
X_train = sc.fit_transform(X_train)
X_train = X_train.reshape(40000, 28, 28, 1)

X_test = train.values[40001:, 1:]
X_test = sc.fit_transform(X_test)
X_test = X_test.reshape(1999, 28, 28, 1)

X_predict = test.values[:28000, :]
X_predict = sc.fit_transform(X_predict)
X_predict = X_predict.reshape(28000, 28, 28, 1)

Y_train = train.values[0:40000, 0]
Y_test = train.values[40001:, 0]

# one hot encoding
Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)

print(train.shape)
print(X_train.shape)
print(Y_train.shape)
print(Y_test.shape)

model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3)

pred = model.predict(X_predict)
test['Label'] = np.argmax(pred, axis=1)
test.index += 1
test.to_csv('mnist_output.csv', columns=['Label'], index_label="ImageId")

