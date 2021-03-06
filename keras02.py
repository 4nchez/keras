from keras.models import Sequential
from keras.layers import Dense

import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x2 = np.array([18,19,20,21,22])

model = Sequential()
model.add(Dense(1000, input_dim=1, activation='relu'))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(98))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x,y, epochs=100)

loss, acc = model.evaluate(x, y)
print('acc : ', acc)
print('loss : ', loss)

y_predict = model.predict(x2)
print(y_predict)