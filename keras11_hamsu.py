#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(1,101))
print(x)

from sklearn.model_selection import train_test_split #test_size, train_size, stratify
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=6, test_size=0.4, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=6, test_size=0.5, shuffle=False)
# 6:2:2

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()
# model.add(Dense(5, input_shape=(1, ), activation='relu'))
# model.add(Dense(10))
# model.add(Dense(5))
# model.add(Dense(1))

# input1 = Input(shape=(1,))
# dense1 = Dense(5,activation='relu')(input1)
# dense2 = Dense(3)(dense1)
# dense3 = Dense(4)(dense2)
# dense3 = Dense(4)(dense2)
# dense4 = Dense(4)(dense3)
# dense5 = Dense(4)(dense4)
# dense6 = Dense(4)(dense5)
# dense7 = Dense(4)(dense6)
# dense8 = Dense(4)(dense7)
# dense9 = Dense(4)(dense8)
# dense10 = Dense(4)(dense9)
# output1 = Dense(1)(dense10)

input1 = Input(shape=(1,))
xx = Dense(5,activation='relu')(input1)
xx = Dense(3)(xx)
xx = Dense(4)(xx)
xx = Dense(4)(xx)
xx = Dense(4)(xx)
xx = Dense(4)(xx)
xx = Dense(4)(xx)
xx = Dense(4)(xx)
output1 = Dense(1)(xx)

model = Model(input = input1, output = output1)
model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam',metrics=['mse'])

# model.fit(x_train,y_train, epochs=100,batch_size=1)
model.fit(x_train,y_train, epochs=100,batch_size=1, validation_data=(x_val, y_val))

#4. 평가예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) # a[0], a[1]
print('mse : ', mse)
print('loss : ', loss)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ",r2_y_predict)

'''
Epoch 100/100
60/60 [==============================] - 0s 715us/step - loss: 1.2889e-11 - mse: 1.2889e-11 - val_loss: 6.1846e-11 - val_mse: 6.1846e-11
20/20 [==============================] - 0s 249us/step
mse :  9.604263911944244e-11
loss :  9.604264050722122e-11
[[ 81.      ]
 [ 81.99999 ]
 [ 82.99999 ]
 [ 84.      ]
 [ 85.      ]
 [ 85.99999 ]
 [ 86.99999 ]
 [ 88.      ]
 [ 89.      ]
 [ 90.00001 ]
 [ 91.000015]
 [ 92.      ]
 [ 93.      ]
 [ 93.99999 ]
 [ 95.      ]
 [ 96.      ]
 [ 97.00001 ]
 [ 98.000015]
 [ 99.00001 ]
 [100.00001 ]]
RMSE:  7.0339542063086605e-06
R2 :  0.999999999998512
'''