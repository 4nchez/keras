#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
x_val = np.array([101,102,103,104,105])
y_val = np.array([121,107,113,4,1225])

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
y_val 수정 전 np.array([101,102,103,104,105])
10/10 [==============================] - 0s 399us/step
mse :  0.008015712723135948
loss :  0.008015713188797235
[[10.958492]
 [11.948805]
 [12.939119]
 [13.929433]
 [14.919746]
 [15.91006 ]
 [16.900373]
 [17.890686]
 [18.880999]
 [19.871311]]
RMSE:  0.08953044005500774
R2 :  0.9990284000367947

y_val 수정 : 102->107
10/10 [==============================] - 0s 296us/step
mse :  0.15895220637321472
loss :  0.15895219817757605
[[10.820192]
 [11.776135]
 [12.732076]
 [13.688017]
 [14.643959]
 [15.5999  ]
 [16.555841]
 [17.511782]
 [18.467724]
 [19.423664]]
RMSE:  0.39868821440683766
R2 :  0.98073305547795

y_val 수정 np.array([121,107,113,4,1225])
10/10 [==============================] - 0s 2ms/step
mse :  0.9012255668640137
loss :  0.901225571334362
[[10.57203 ]
 [11.46709 ]
 [12.362149]
 [13.257206]
 [14.152265]
 [15.047323]
 [15.942383]
 [16.837439]
 [17.7325  ]
 [18.62756 ]]
RMSE:  0.9493286221313161
R2 :  0.8907606263275463
'''