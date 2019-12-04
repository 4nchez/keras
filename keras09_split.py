#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(1,101))
print(x)

x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]
y_train = x[:60]
y_test = x[60:80]
y_val = x[80:]

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
20/20 [==============================] - 0s 248us/step
mse :  399.999755859375
loss :  399.9997863769531
[[80.99999 ]
 [81.99999 ]
 [82.99999 ]
 [83.99999 ]
 [84.999985]
 [85.99999 ]
 [87.      ]
 [87.99999 ]
 [89.      ]
 [89.999985]
 [90.99999 ]
 [91.999985]
 [92.99999 ]
 [93.999985]
 [94.99999 ]
 [95.99999 ]
 [96.99999 ]
 [97.999985]
 [98.99999 ]
 [99.99998 ]]
RMSE:  19.999990463257546
R2 :  -11.030063715199782
'''