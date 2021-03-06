#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(1,101))
print(x)

from sklearn.model_selection import train_test_split #test_size, train_size, stratify
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=6, train_size=0.6, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=6, test_size=0.5, shuffle=False)
# 6:2:2

#2. 모델구성
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
model = load_model('./save/savetest01.h5')
model.add(Dense(100, name='dense_13'))
model.add(Dense(1, name='dense_111'))
model.summary()


#3. 훈련
model.compile(loss='mse', optimizer='adam',metrics=['mse'])
import keras
tb_hist = keras.callbacks.TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
from keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train,y_train, epochs=100,batch_size=1, validation_data=(x_val, y_val), callbacks=[early_stopping, tb_hist])

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
