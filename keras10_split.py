#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(1,101))
print(x)

# x_train = x[:60]
# x_val = x[60:80]
# x_test = x[80:]
# y_train = x[:60]
# y_test = x[60:80]
# y_val = x[80:]

from sklearn.model_selection import train_test_split #test_size, train_size, stratify
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=6, train_size=0.9, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=6, test_size=0.5, shuffle=False)
# 6:2:2

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(256, input_dim=1, activation='relu'))
model.add(Dense(5, input_shape=(1, ), activation='relu'))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

# model.summary()

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
train 60, test 20, val 20,
20/20 [==============================] - 0s 298us/step
mse :  5.625393129093936e-08
loss :  5.625392986985389e-08
[[52.999912 ]
 [58.99987  ]
 [99.999565 ]
 [41.999996 ]
 [93.99962  ]
 [83.99968  ]
 [ 6.0002594]
 [24.000128 ]
 [48.999947 ]
 [54.9999   ]
 [70.99979  ]
 [ 4.0002737]
 [16.000187 ]
 [45.999966 ]
 [68.999794 ]
 [66.99981  ]
 [88.99965  ]
 [27.000101 ]
 [87.99966  ]
 [82.999695 ]]
RMSE:  0.00023853995094863222
R2 :  0.9999999999312462

train 60, test 20, val 20, shuffle=False
20/20 [==============================] - 0s 250us/step
mse :  4.3655745685100555e-11
loss :  4.3655745685100555e-11
[[ 81.      ]
 [ 82.      ]
 [ 82.99999 ]
 [ 84.00001 ]
 [ 85.      ]
 [ 85.99999 ]
 [ 87.      ]
 [ 88.      ]
 [ 89.      ]
 [ 90.      ]
 [ 91.      ]
 [ 91.99999 ]
 [ 92.999985]
 [ 93.99999 ]
 [ 94.999985]
 [ 96.      ]
 [ 97.      ]
 [ 97.99999 ]
 [ 99.      ]
 [100.      ]]
RMSE:  6.383209430954556e-06
R2 :  0.9999999999987745

train 90, test 5, val 5, shuffle=False
90/90 [==============================] - 0s 632us/step - loss: 2.0174e-07 - mse: 2.0174e-07 - val_loss: 1.1711e-07 - val_mse: 1.1711e-07
5/5 [==============================] - 0s 192us/step
mse :  1.352629652728865e-07
loss :  1.3526296243071557e-07
[[95.99964 ]
 [96.99964 ]
 [97.99963 ]
 [98.999626]
 [99.999626]]
RMSE:  0.0003662745102697698
R2 :  0.9999999329214916
'''