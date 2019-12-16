# x, y 2개씩 분리 = input 2 output 2

from numpy import array
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
from sklearn.model_selection import train_test_split #test_size, train_size, stratify
x_1, x_2, y_1, y_2 = train_test_split(x, y, random_state=6, train_size=0.5, shuffle=False)
print(x_1.shape) #(6,3)
print(x_2.shape) #(7,3)
print(y_1.shape) #(6, )
print(y_2.shape) #(7, )
# # x_1 = x_1.reshape((6,3,1))
x_1 = x_1.reshape((x_1.shape[0], x_1.shape[1], 1))
x_2 = x_2.reshape((x_2.shape[0], x_2.shape[1], 1))
print(x_1.shape) #(6,3,1)
print(x_2.shape) #(7,3,1)

# 2. 모델 구성
input1 = Input(batch_shape=(None,3,1))
dense1 = LSTM(15,activation='relu')(input1)
dense2 = Dense(15)(dense1)
middle1 = Dense(15)(dense2)

input2 = Input(batch_shape=(None,3,1))
xx = LSTM(15,activation='relu')(input2)
xx = Dense(15)(xx)
middle2 = Dense(15)(xx)

# concatenate
from keras.layers.merge import concatenate
marge1 = concatenate([middle1, middle2])
output1 = Dense(15)(marge1)
output1 = Dense(15)(output1)
output1 = Dense(1)(output1)

output2 = Dense(15)(marge1)
output2 = Dense(15)(output2)
output2 = Dense(1)(output2)

model = Model(input = [input1, input2], output = [output1, output2])
model.summary()

# 3. 실행
model.compile(optimizer='adam', loss='mse')
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit([x_1, x_2], [y_1, y_2], epochs=1000, batch_size=1, callbacks=[early_stopping])

# predict용 데이터
x_input = array([25,35,45])
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)