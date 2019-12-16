from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = array([4,5,6,7])
print(x)
print('x.shape : ', x.shape) # (4, 3)
print('y.shape : ', y.shape) # (4, )

'''
 x  y
123 4
234 5
345 6
456 7
'''

x = x.reshape((x.shape[0], x.shape[1], 1))
print("x.shape : ",x.shape) # (4, 3, 1) 1: 자르는 개수
''' x 
[1],[2],[3]
'''
print(x)
# 2. 모델 구성
model = Sequential()
model.add(LSTM(51, activation='relu', input_shape=(3,1)))#3 : 열 1: 자르는 개수
model.add(Dense(35))
model.add(Dense(14))
model.add(Dense(1))
#자체제작
# xInput = Input(shape=(3, 1))
# xLstm_1 = LSTM(60, return_sequences=True)(xInput)
# xLstm_1 = LSTM(50, return_sequences=True)(xLstm_1)
# xLstm_2 = LSTM(60, return_sequences=True)(xLstm_1)
# xLstm_2 = LSTM(50)(xLstm_1)
# xOutput = Dense(1)(xLstm_2)

# model.summary()

# 3. 실행
# model = Model(xInput,xOutput)
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=500, batch_size=1)

x_input = array([6,7,8]) #(1, 3)
x_input = x_input.reshape((1,3,1)) #(1, 3, 1)

yhat = model.predict(x_input)
print(yhat)
'''
[[9.014222]]
[[8.990221]]
[[9.053764]]
'''