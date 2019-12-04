from keras.models import Sequential
from keras.layers import Dense

import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
x_predict = np.array([21,22,23,24,25])

model = Sequential()
# model.add(Dense(256, input_dim=1, activation='relu'))
model.add(Dense(256, input_shape=(1, ), activation='relu'))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=100,batch_size=1)

loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print('acc : ', acc)
print('loss : ', loss)

y_predict = model.predict(x_predict)
print(y_predict)
