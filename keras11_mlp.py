#1. 데이터
import numpy as np
x = np.array([range(1,101), range(101,201)])
y = np.array([range(1,101), range(101,201)])
# print(x)
print(x.shape)

x = np.transpose(x)
y = np.transpose(y)
# print(x)
print(x.shape)

# x_train = x[:60]
# x_val = x[60:80]
# x_test = x[80:]
# y_train = x[:60]
# y_test = x[60:80]
# y_val = x[80:]

from sklearn.model_selection import train_test_split #test_size, train_size, stratify
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=6, train_size=0.6, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=6, test_size=0.5, shuffle=False)
# 6:2:2

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(256, input_dim=1, activation='relu'))
model.add(Dense(5, input_shape=(2, ), activation='relu'))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(2))

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
Epoch 100/100
60/60 [==============================] - 0s 980us/step - loss: 6.2010e-11 - mse: 6.2010e-11 - val_loss: 1.0768e-10 - val_mse: 1.0768e-10
20/20 [==============================] - 0s 300us/step
mse :  1.4842953255378433e-10
loss :  1.4842953532934188e-10
[[ 81.00001 181.     ]
 [ 81.99999 182.00002]
 [ 83.00001 183.00003]
 [ 83.99999 184.00002]
 [ 84.99999 185.     ]
 [ 86.      186.00002]
 [ 86.99999 187.00002]
 [ 88.      188.00003]
 [ 89.      189.00002]
 [ 89.99999 190.     ]
 [ 91.      191.00002]
 [ 92.      192.00002]
 [ 92.99999 193.00002]
 [ 94.      194.00002]
 [ 94.99999 195.00002]
 [ 95.99999 196.     ]
 [ 96.99999 197.00002]
 [ 97.99999 198.00002]
 [ 98.99999 199.00002]
 [ 99.99999 200.00002]]
RMSE:  1.2183166063439418e-05
R2 :  0.999999999995536
'''