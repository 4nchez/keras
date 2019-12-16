#input 1 output2
#1. 데이터
import numpy as np

x1 = np.array([range(100), range(311,411), range(100)])

y1 = np.array([range(100,200), range(311,411), range(100,200)])
y2 = np.array([range(501,601), range(711,811), range(100)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

print(x1.shape) # (100,3)
print(y1.shape) # (100,3)
print(y2.shape) # (100,3)

from sklearn.model_selection import train_test_split #test_size, train_size, stratify
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=6, test_size=0.4, shuffle=False)
x1_val, x1_test, y1_val, y1_test = train_test_split(x1_test, y1_test, random_state=6, test_size=0.5, shuffle=False)
y2_train, y2_test = train_test_split(y2, random_state=6, test_size=0.4, shuffle=False)
y2_val, y2_test = train_test_split(y2_test, random_state=6, test_size=0.5, shuffle=False)
# 6:2:2
print(y2_test.shape) # (20,3)

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

# concatenate
# from keras.layers.merge import concatenate
output1 = Dense(31)(middle1)
output1 = Dense(32)(output1)
output1 = Dense(3)(output1)

output2 = Dense(31)(middle1)
output2 = Dense(32)(output2)
output2 = Dense(3)(output2)
model = Model(input = input1, output = [output1,output2])
model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam',metrics=['mse'])

model.fit(x1_train, [y1_train,y2_train], epochs=100, batch_size=1, validation_data=(x1_val,[y1_val,y2_val]))

#4. 평가예측
mse = model.evaluate(x1_train,[y1_train,y2_train], batch_size=1) # a[0], a[1]
print('mse : ', mse)
# print('mse[0] : ', mse[0])
# print('mse[1] : ', mse[1])
# print('mse[2] : ', mse[2])
# print('mse[3] : ', mse[3])
# print('mse[4] : ', mse[4]) # x1, merge, y1,y2 =4개

y1_predict, y2_predict = model.predict(x1_test)
print(y1_predict, y2_predict)
# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE(1): ", RMSE1)
print("RMSE(2): ", RMSE2)
print("RMSE: ", (RMSE1 + RMSE2)/2)

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y1_test, y1_predict)
r2_y_predict2 = r2_score(y2_test, y2_predict)
print("R2(1) : ",r2_y_predict)
print("R2(2) : ",r2_y_predict2)
print("R2 : ",(r2_y_predict + r2_y_predict2)/2)
'''
Epoch 100/100
 1/60 [..............................] - ETA: 0s - loss: 1.7177 - dense_8_loss: 0.3895 - dense_11_loss: 1.3282 -51/60 [========================>.....] - ETA: 0s - loss: 0.9057 - dense_8_loss: 0.1909 - dense_11_loss: 0.7148 -60/60 [==============================] - 0s 1ms/step - loss: 0.8296 - dense_8_loss: 0.1754 - dense_11_loss: 0.6542 - dense_8_mse: 0.1754 - dense_11_mse: 0.6542 - val_loss: 0.1148 - val_dense_8_loss: 0.0230 - val_dense_11_loss: 0.0918 - val_dense_8_mse: 0.0230 - val_dense_11_mse: 0.0918
60/60 [==============================] - 0s 382us/step
mse :  [0.3178746533890565, 0.06935053318738937, 0.24852411448955536, 0.06935053318738937, 0.24852411448955536]
[[179.98543 391.2011  179.93439]
 [180.98582 392.1968  180.93614]
 [181.98622 393.1924  181.93796]
 [182.9867  394.18817 182.93974]
 [183.9871  395.18384 183.94153]
 [184.98746 396.17957 184.94327]
 [185.98795 397.1752  185.94507]
 [186.98834 398.1708  186.94684]
 [187.9888  399.16647 187.94865]
 [188.9892  400.16223 188.95042]
 [189.98964 401.15793 189.95226]
 [190.99002 402.15366 190.95404]
 [191.99042 403.14932 191.95584]
 [192.9909  404.14496 192.95766]
 [193.99132 405.14062 193.95938]
 [194.99173 406.13623 194.9612 ]
 [195.99217 407.132   195.963  ]
 [196.9926  408.12766 196.9648 ]
 [197.99298 409.12335 197.96658]
 [198.99342 410.1191  198.9683 ]] [[581.2434   791.3482    79.89705 ]
 [582.2398   792.34143   80.90025 ]
 [583.2361   793.3347    81.90347 ]
 [584.2325   794.32794   82.90671 ]
 [585.2288   795.3212    83.90988 ]
 [586.2251   796.31445   84.91306 ]
 [587.22156  797.3078    85.91623 ]
 [588.2178   798.30096   86.91943 ]
 [589.2141   799.2943    87.9227  ]
 [590.2106   800.28754   88.92578 ]
 [591.207    801.28094   89.929054]
 [592.2033   802.2743    90.93224 ]
 [593.1996   803.2675    91.935455]
 [594.196    804.26074   92.93865 ]
 [595.19226  805.2538    93.941826]
 [596.18866  806.24713   94.945045]
 [597.1851   807.2407    95.94823 ]
 [598.18134  808.23395   96.95148 ]
 [599.17786  809.2273    97.954636]
 [600.17413  810.2204    98.95786 ]]
RMSE(1):  0.09803808583786187
RMSE(2):  0.2097066943243852
RMSE:  0.15387239008112355
R2(1) :  0.9997109333451202
R2(2) :  0.9986773865370088
R2 :  0.9991941599410645
'''