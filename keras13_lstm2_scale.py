from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
# print(x)
print("x.shape : ",x.shape) #(13, 3)
print("y.shape : ",y.shape) #(13, )
x = x.reshape((13,3,1))
print("x.shape : ",x.shape)

# 2. 모델 구성
model = Sequential()
model.add(LSTM(5, activation='relu', input_shape=(3,1)))#3 : 열 1: 자르는 개수
model.add(Dense(3))
model.add(Dense(1))

# 3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=1000, batch_size=1)

# predict용 데이터
x_input = array([25,35,45])
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)
