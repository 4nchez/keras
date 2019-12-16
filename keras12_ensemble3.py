#input 2 output1
#1. 데이터
import numpy as np

x1 = np.array([range(100), range(311,411), range(100)])
x2 = np.array([range(501,601), range(711,811), range(100)])

y1 = np.array([range(100,200), range(311,411), range(100,200)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)

print(x1.shape) # (100,3)
print(x2.shape) # (100,3)
print(y1.shape) # (100,3)

from sklearn.model_selection import train_test_split #test_size, train_size, stratify
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=6, test_size=0.4, shuffle=False)
x1_val, x1_test, y1_val, y1_test = train_test_split(x1_test, y1_test, random_state=6, test_size=0.5, shuffle=False)
x2_train, x2_test = train_test_split(x2, random_state=6, test_size=0.4, shuffle=False)
x2_val, x2_test = train_test_split(x2_test, random_state=6, test_size=0.5, shuffle=False)
# 6:2:2
print(x2_test.shape) # (20,3)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()
# model.add(Dense(5, input_shape=(1, ), activation='relu'))
# model.add(Dense(10))
# model.add(Dense(5))
# model.add(Dense(1))

input1 = Input(shape=(3,))
dense1 = Dense(5,activation='relu')(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
dense4 = Dense(4)(dense3)
middle1 = Dense(3)(dense4)

input2 = Input(shape=(3,))
xx = Dense(5,activation='relu')(input2)
xx = Dense(3)(xx)
xx = Dense(4)(xx)
xx = Dense(4)(xx)
middle2 = Dense(3)(xx)

# concatenate
from keras.layers.merge import concatenate
marge1 = concatenate([middle1, middle2])
output1 = Dense(31)(marge1)
output1 = Dense(32)(output1)
output1 = Dense(3)(output1)

model = Model(input = [input1, input2], output = output1)
model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam',metrics=['mse'])

# model.fit(x_train,y_train, epochs=100,batch_size=1)
# model.fit(x1_train,y1_train, epochs=100,batch_size=1, validation_data=(x1_val, y1_val))
model.fit([x1_train,x2_train],y1_train, epochs=100, batch_size=1, validation_data=([x1_val, x2_val],y1_val))

#4. 평가예측
mse = model.evaluate([x1_train,x2_train],y1_train, batch_size=1) # a[0], a[1]
print('mse : ', mse) # x1, x2, merge, y1 =4개

y1_predict = model.predict([x1_test,x2_test])
print(y1_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
    # return np.sqrt(mean_squared_error(y1_test, y1_predict))
# print("RMSE: ", RMSE(y1_test, y1_predict))
# def RMSE2(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y2_test, y2_predict))
# print("RMSE2: ", RMSE2(y2_test, y2_predict))
def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))
RMSE1 = RMSE(y1_test, y1_predict)
print("RMSE(1): ", RMSE1)

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y1_test, y1_predict)
print("R2(1) : ",r2_y_predict)

'''
Epoch 100/100
60/60 [==============================] - 0s 1ms/step - loss: 0.0133 - mse: 0.0133 - val_loss: 0.0034 - val_mse:
0.0034
60/60 [==============================] - 0s 316us/step
mse :  [0.0022047838952858, 0.002204783959314227]
[[179.95418 390.91562 180.04727]
 [180.95366 391.9155  181.04784]
 [181.95311 392.9152  182.04834]
 [182.95262 393.91504 183.04887]
 [183.9521  394.91483 184.04945]
 [184.9516  395.9147  185.04999]
 [185.95103 396.91455 186.05054]
 [186.95056 397.91434 187.05113]
 [187.95004 398.91418 188.0516 ]
 [188.94951 399.91388 189.05217]
 [189.94902 400.91373 190.05267]
 [190.94847 401.9137  191.05324]
 [191.948   402.91345 192.05376]
 [192.94746 403.9132  193.05423]
 [193.94695 404.913   194.05487]
 [194.94646 405.9129  195.05533]
 [195.94588 406.9126  196.05588]
 [196.94543 407.91248 197.05638]
 [197.94487 408.9123  198.05705]
 [198.9444  409.9121  199.05756]]
RMSE(1):  0.06521615724701291
R2(1) :  0.9998720857995168
'''