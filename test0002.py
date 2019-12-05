import numpy as np

aaa = np.array([1,2,3,4,5])
print(aaa.shape)

aaa = aaa.reshape(5,1)
print(aaa.shape)

bbb = np.array([[1,2,3], [4,5,6]])
print(bbb.shape)

ccc = np.array([[1,2], [3,4], [5,6]])
print(ccc.shape)

ddd = ccc.reshape(3,1,2,1)
print(ddd.shape)
print(ddd)