from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(X_train[0])
print(Y_test[0])
print(X_train.shape) # (60000, 28, 28)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import tensorflow as tf

X_train = X_train.reshape(X_train.shape[0], 28*28, 1).astype('float32') / 255 # MAX 값으로 나눔 | MinMaxScaler
X_test = X_test.reshape(X_test.shape[0], 28*28, 1).astype('float32') / 255 # MAX 값으로 나눔 |MinMaxScaler
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
print(Y_train[0])
print(Y_train.shape)
print(Y_test.shape)

# 컨볼루션 신경망의 설정
model = Sequential() 
model.add(LSTM(32, input_shape=(28*28, 1), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlyStopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test),
                    epochs=2, batch_size=200, verbose=1, callbacks=[earlyStopping_callback])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" %(model.evaluate(X_test, Y_test)[1]))
# / 255 | Test Accuracy: 0.0980