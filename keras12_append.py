#1. 데이터
import numpy as np

x1 = np.array([range(100), range(311,411), range(100)])
y1 = np.array([range(501,601), range(711,811), range(100)])

x2 = np.array([range(100,200), range(311,411), range(100,200)])
y2 = np.array([range(501,601), range(711,811), range(100)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)

print(x1.shape) # (100,3)
print(x2.shape) # (100,3)
print(y1.shape) # (100,3)
print(y2.shape) # (100,3)

x = np.hstack([x1, x2])
y = np.hstack([y1, y2])
print(x.shape) # (100,6)
print(y.shape) # (100,6)
# print(x)

from sklearn.model_selection import train_test_split #test_size, train_size, stratify
# x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=6, test_size=0.4, shuffle=False)
# x1_val, x1_test, y1_val, y1_test = train_test_split(x1_test, y1_test, random_state=6, test_size=0.5, shuffle=False)
# x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=6, test_size=0.4, shuffle=False)
# x2_val, x2_test, y2_val, y2_test = train_test_split(x2_test, y2_test, random_state=6, test_size=0.5, shuffle=False)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=6, test_size=0.4, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=6, test_size=0.5, shuffle=False)
# 6:2:2
print(x_test.shape) # (20,6)


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()
# model.add(Dense(5, input_shape=(1, ), activation='relu'))
# model.add(Dense(10))
# model.add(Dense(5))
# model.add(Dense(1))

input1 = Input(shape=(6,))
dense1 = Dense(5,activation='relu')(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
dense3 = Dense(4)(dense2)
dense4 = Dense(4)(dense3)
middle1 = Dense(6)(dense4)

# input2 = Input(shape=(6,))
# xx = Dense(5,activation='relu')(input2)
# xx = Dense(3)(xx)
# xx = Dense(4)(xx)
# xx = Dense(4)(xx)
# xx = Dense(4)(xx)
# xx = Dense(4)(xx)
# xx = Dense(4)(xx)
# xx = Dense(4)(xx)
# middle2 = Dense(6)(xx)

# concatenate
# from keras.layers.merge import concatenate
# marge1 = concatenate([middle1, middle2])
# output1 = Dense(30)(marge1)
# output1 = Dense(32)(output1)
# output1 = Dense(6)(output1)

# output2 = Dense(30)(marge1)
# output2 = Dense(13)(output2)
# output2 = Dense(6)(output2)

# model = Model(input = [input1, input2], output = [output1, output2])
model = Model(input = input1, output = middle1)
model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam',metrics=['mse'])

# model.fit(x_train,y_train, epochs=100,batch_size=1)
model.fit(x_train,y_train, epochs=550,batch_size=1, validation_data=(x_val, y_val))
# model.fit(x1_train,y1_train,x2_train,y2_train, epochs=100,batch_size=1, validation_data=(x1_val, y1_val,x2_val, y2_val))


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
Epoch 550/550
60/60 [==============================] - 0s 1ms/step - loss: 1.0461e-06 - mse: 1.0461e-06 - val_loss: 1.3489e-05 - val_mse:
1.3489e-05
20/20 [==============================] - 0s 251us/step
mse :  1.7176051187561825e-05
loss :  1.717605059639027e-05
[[581.0033   791.00446   80.00014  581.003    791.0047    80.00573 ]
 [582.00336  792.0044    81.00012  582.00305  792.00464   81.0056  ]
 [583.00336  793.0045    82.00014  583.00305  793.0047    82.00571 ]
 [584.0034   794.00446   83.00013  584.0031   794.0047    83.00575 ]
 [585.0035   795.0046    84.000046 585.0031   795.00476   84.00569 ]
 [586.0034   796.00446   85.0001   586.0031   796.00464   85.00574 ]
 [587.0035   797.0045    86.00005  587.0032   797.0047    86.00578 ]
 [588.0035   798.0046    87.00005  588.0031   798.0047    87.00589 ]
 [589.0035   799.00446   88.00007  589.0032   799.0047    88.005875]
 [590.00354  800.00464   89.00007  590.00323  800.0048    89.00593 ]
 [591.0036   801.00464   90.00004  591.0032   801.00476   90.00592 ]
 [592.00366  802.0047    91.00002  592.0033   802.0049    91.005974]
 [593.00366  803.0047    92.00002  593.00336  803.0048    92.00602 ]
 [594.00366  804.0047    93.       594.0034   804.00494   93.006065]
 [595.00366  805.0047    94.       595.0033   805.0048    94.00605 ]
 [596.0037   806.00476   94.999985 596.00336  806.0048    95.0061  ]
 [597.0037   807.00476   95.999954 597.00336  807.0049    96.006096]
 [598.0037   808.00476   96.99997  598.0034   808.00494   97.0062  ]
 [599.00385  809.0049    97.999954 599.0035   809.00507   98.006195]
 [600.00385  810.0049    98.99994  600.0035   810.005     99.00618 ]]
RMSE:  0.004141385053001134
R2 :  0.9999994841783412
'''