from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import tensorflow as tf

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

X1_train = X_train[:30000]
X2_train = X_train[30000:]
X1_test = X_test[5000:]
X2_test = X_test[:5000]

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

Y1_train = Y_train[:30000]
Y2_train = Y_train[30000:]
Y1_test = Y_test[5000:]
Y2_test = Y_test[:5000]

print(X1_train.shape) # (30000, 28, 28, 1)
print(X2_train.shape) # (30000, 28, 28, 1)
print(X1_test.shape)
print(Y1_test.shape)
print(Y1_train.shape)
print(Y2_train.shape)

# 컨볼루션 신경망의 설정
input1 = Input(shape=(28, 28, 1))
d = Conv2D(32, kernel_size=(3,3), activation='relu')(input1)
d = Conv2D(64, (3,3),  activation='relu')(d)
d = MaxPooling2D(pool_size=2)(d)
d = Flatten()(d)
middle1 = Dense(10)(d)

input2 = Input(shape=(28, 28, 1))
xx = Conv2D(32, kernel_size=(3,3), activation='relu')(input2)
xx = Conv2D(64, (3,3),  activation='relu')(xx)
xx = MaxPooling2D(pool_size=2)(xx)
xx = Flatten()(xx)
middle2 = Dense(10)(xx)

# concatenate
from keras.layers.merge import concatenate
marge1 = concatenate([middle1, middle2])
output1 = Dense(10)(marge1)
output1 = Dense(10,activation='softmax')(output1)

output2 = Dense(10)(marge1)
output2 = Dense(10,activation='softmax')(output2)

model = Model(input = [input1, input2], output = [output1, output2])
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlyStopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 실행
history = model.fit([X1_train,X2_train], [Y1_train,Y2_train], validation_data=([X1_test,X2_test], [Y1_test,Y2_test]),
                    epochs=2, batch_size=200, verbose=1, callbacks=[earlyStopping_callback]) #epochs=30

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" %(model.evaluate([X1_test,X2_test], [Y1_test,Y2_test])[1]))

# Test Accuracy: 0.0960