import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,11))

size = 5
def split_x(seq, size):
    aaa =[]
    for i in range(len(a)- size + 1):
        subset = a[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print("====================")
print(dataset)
print(dataset.shape)

x_train = dataset[:,0:4]
y_train = dataset[:,4] 

print(x_train.shape) #(6,4)
print(y_train.shape) #(6, )

model = Sequential()
model.add(Dense(13,input_shape=(4, )))
model.add(Dense(4))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam',metrics=['mse'])
model.fit(x_train,y_train, epochs=100, batch_size=1)

x2 = np.array([7,8,9,10])
x2 = x2.reshape((1,4))
y_pred = model.predict(x2)
print(y_pred)