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

from sklearn.model_selection import train_test_split #test_size, train_size, stratify
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=6, test_size=0.4, shuffle=False)
x1_val, x1_test, y1_val, y1_test = train_test_split(x1_test, y1_test, random_state=6, test_size=0.5, shuffle=False)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=6, test_size=0.4, shuffle=False)
x2_val, x2_test, y2_val, y2_test = train_test_split(x2_test, y2_test, random_state=6, test_size=0.5, shuffle=False)
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
dense1 = Dense(150,activation='relu')(input1)
dense2 = Dense(150)(dense1)
dense3 = Dense(150)(dense2)
dense3 = Dense(150)(dense2)
dense4 = Dense(150)(dense3)
middle1 = Dense(150)(dense4)

input2 = Input(shape=(3,))
xx = Dense(150,activation='relu')(input2)
xx = Dense(150)(xx)
xx = Dense(150)(xx)
xx = Dense(150)(xx)
xx = Dense(150)(xx)
middle2 = Dense(150)(xx)

# concatenate
from keras.layers.merge import concatenate
marge1 = concatenate([middle1, middle2])
output1 = Dense(15)(marge1)
output1 = Dense(15)(output1)
output1 = Dense(3)(output1)

output2 = Dense(15)(marge1)
output2 = Dense(15)(output2)
output2 = Dense(3)(output2)

model = Model(input = [input1, input2], output = [output1, output2])
model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam',metrics=['mse'])

# model.fit(x_train,y_train, epochs=100,batch_size=1)
# model.fit(x1_train,y1_train, epochs=100,batch_size=1, validation_data=(x1_val, y1_val))
model.fit([x1_train,x2_train],[y1_train,y2_train], epochs=100,batch_size=1, validation_data=([x1_val, x2_val],[y1_val, y2_val]))

#4. 평가예측
mse = model.evaluate([x1_train,x2_train],[y1_train,y2_train], batch_size=1) # a[0], a[1]
print('mse[0] : ', mse[0])
print('mse[1] : ', mse[1])
print('mse[2] : ', mse[2])
print('mse[3] : ', mse[3])
print('mse[4] : ', mse[4]) # x1, x2, merge, y1, y2 =5개

y1_predict, y2_predict = model.predict([x1_test,x2_test])
print(y1_predict, y2_predict)

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
60/60 [==============================] - 0s 399us/step
mse[0] :  0.8926110888520876
mse[1] :  0.43203017115592957
mse[2] :  0.4605810046195984
mse[3] :  0.43203017115592957
mse[4] :  0.4605810046195984
RMSE(1):  4.778738820977907
RMSE(2):  5.150435805685485
RMSE:  4.9645873133316964
R2(1) :  0.3131926400264267
R2(2) :  0.20219581989512517
R2 :  0.25769422996077596

데이터가 별로라 R2값도 별로다
'''