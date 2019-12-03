from keras.models import Sequential
from keras.layers import Dense

import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
# x_predict = np.array([21,22,23,24,25])

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

model.compile(loss='mse', optimizer='adam', #metrics=['accuracy']
                                            metrics=['mse'])
model.fit(x_train,y_train, epochs=100,batch_size=1)

loss, mse = model.evaluate(x_test, y_test, batch_size=1) # a[0], a[1]
print('mse : ', mse) #1.0 | 1.6484572995523195e-07
print('loss : ', loss) #0.000604250532342121 | 1.6484573279740288e-07

'''
y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))
'''