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
model.add(Dense(100, input_shape=(2, ), activation='relu'))
model.add(Dense(200))
model.add(Dense(400))
model.add(Dense(500))
model.add(Dense(700))
model.add(Dense(100))
model.add(Dense(1))

# model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam',metrics=['mse'])

# model.fit(x_train,y_train, epochs=100,batch_size=1)
model.fit(x_train,y_train, epochs=101,batch_size=1, validation_data=(x_val, y_val))

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
Epoch 101/101
60/60 [==============================] - 0s 8ms/step - loss: 22.4472 - mse: 22.4472 - val_loss:
217.5681 - val_mse: 217.5681
20/20 [==============================] - 0s 449us/step
mse :  442.694091796875
loss :  442.69408416748047
[[298.89517]
 [300.2176 ]
 [301.54004]
 [302.86237]
 [304.18475]
 [305.5072 ]
 [306.82962]
 [308.1521 ]
 [309.47427]
 [310.7969 ]
 [312.11914]
 [313.44156]
 [314.76407]
 [316.08643]
 [317.4089 ]
 [318.73117]
 [320.05365]
 [321.37598]
 [322.69833]
 [324.02087]]
RMSE:  21.040297688480212
R2 :  -12.314109077289196
'''