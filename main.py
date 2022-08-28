import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import _pickle as pickle
import matplotlib.pyplot as plt
mnist = "D:/Users/lenovo/HWDG"
x,test_x,y,test_y = train_test_split(mnist.data,mnist.target,test_size=0.25,random_state=40)
model = svm.LinearSVC()
model.fit(x,y)
z=model.predict(test_x)
print('准确率：',np.sum(z==test_y)/z.size)
with open('D:/shujukexue/model.pkl', 'wb') as file:
    pickle.dump(model, file)
# 学习后识别520到525六张图片并给出预测
model.predict(mnist.data[520:526])
# 实际的520到525代表的数
mnist.target[520:526]
# 显示520到525数字图片
plt.subplot(321)
plt.imshow(mnist.images[520], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(322)
plt.imshow(mnist.images[521], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(323)
plt.imshow(mnist.images[522], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(324)
plt.imshow(mnist.images[523], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(325)
plt.imshow(mnist.images[524], cmap=plt.cm.gray_r, interpolation='nearest')
plt.subplot(326)
plt.imshow(mnist.images[525], cmap=plt.cm.gray_r, interpolation='nearest')