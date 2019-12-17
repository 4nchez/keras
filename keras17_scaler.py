from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20000,30000,40000], [30000,40000,50000], [40000,50000,60000],
            [100,200,300]])
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400])

from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x)
x = scaler.transform(x) # evaluate, predict
print(x)

# train 과 predict로 나눈다 train(1, 14), predict(14)
x_train = x[:13] #(13, 3)
x_predict = x[13:] #(1 , 3)
y_train = y[:13] #(13,  )
y_predict = y[13:] #(1 ,  )

print(x_train.shape, x_predict.shape)
print(y_train.shape, y_predict.shape)

# 2. 모델 구성
model = Sequential()
model.add(Dense(51, activation='relu', input_shape=(3, )))
model.add(Dense(35))
model.add(Dense(14))
model.add(Dense(1))

# 3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)

# 4. 예측
yhat = model.predict(x_predict)
print(yhat)