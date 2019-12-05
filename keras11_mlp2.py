#1. 데이터
import numpy as np
x = np.array([range(1,101), range(101,201)])
y = np.array([range(201,301)])
# print(x)
print(x.shape) #(2,100)

x = np.transpose(x)
y = np.transpose(y)

print(x.shape) #(100,2)

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

# aaa = np.array([[101,102,103], [201,202,203]])
# aaa = np.transpose(aaa)
# y_predict = model.predict(aaa)
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
60/60 [==============================] - 0s 798us/step - loss: 4.3850e-10 - mse: 4.3850e-10 - val_loss: 3.0268e-09 - val_mse: 3.0268e-09
20/20 [==============================] - 0s 326us/step
mse :  5.774199784980283e-09
loss :  5.774199962615967e-09
[[281.00006]
 [282.00006]
 [283.00006]
 [284.0001 ]
 [285.00006]
 [286.00006]
 [287.0001 ]
 [288.00006]
 [289.00006]
 [290.00006]
 [291.00006]
 [292.0001 ]
 [293.00006]
 [294.00006]
 [295.0001 ]
 [296.0001 ]
 [297.00006]
 [298.0001 ]
 [299.0001 ]
 [300.0001 ]]
RMSE:  7.475249459177179e-05
R2 :  0.9999999998319418

aaa
60/60 [==============================] - 0s 765us/step - loss: 3.1820e-10 - mse: 3.1820e-10 - val_loss: 3.7253e-10 - val_mse: 3.7253e-10
20/20 [==============================] - 0s 301us/step
mse :  1.0710209386033398e-09
loss :  1.0710209608078002e-09
[[301.     ]
 [301.99997]
 [302.99997]]
'''