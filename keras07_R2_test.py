# 문제 1. R2 0.5 이하로 줄이시오.
# 레이어는 인풋과 아웃풋 포함 5개 이상, 노드는 각 레이어당 5개 이상
# batch_size = 1
# epochs = 100 이상

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
model.add(Dense(600, input_shape=(1, ), activation='relu'))
model.add(Dense(600))
model.add(Dense(600))
model.add(Dense(600))
model.add(Dense(800))
model.add(Dense(100))
model.add(Dense(500))
model.add(Dense(200))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', #metrics=['accuracy']
                                            metrics=['mse'])
model.fit(x_train,y_train, epochs=100,batch_size=1)

loss, mse = model.evaluate(x_test, y_test, batch_size=1) # a[0], a[1]
print('mse : ', mse) #1.0 | 1.6484572995523195e-07
print('loss : ', loss) #0.000604250532342121 | 1.6484573279740288e-07


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
10/10 [==============================] - 0s 23ms/step - loss: 8.7033 - mse: 8.7033
10/10 [==============================] - 0s 2ms/step
mse :  109.8917465209961
loss :  109.89176082611084
[[5.3949904]
 [5.3965006]
 [5.39801  ]
 [5.399522 ]
 [5.401027 ]
 [5.4096885]
 [5.4165974]
 [5.4210377]
 [5.436531 ]
 [5.4580407]]
RMSE:  10.482926942912727
R2 :  -12.320213004902493
'''