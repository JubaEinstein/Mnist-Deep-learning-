'''
Author: @Juba.agoun 
'''

import numpy as np
from keras.datasets import mnist
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



(x_train, y_train), (x_test, y_test) = mnist.load_data()



nb_images=10000
early_stopping_monitor= EarlyStopping(patience=2)



X=np.empty((0,784))
for i,x in enumerate(x_train[0:nb_images]):
X=np.append(X,np.matrix(x.flatten()),axis=0)



model=Sequential()
model.add(Dense(32,activation='relu',input_shape=(X.shape[1],)))
model.add(Dense(50,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

target = keras.utils.to_categorical(y_train[0:nb_images])
model.fit(X,target,validation_split=0.3,nb_epoch=20,callbacks=[early_stopping_monitor])




%matplotlib inline

index_image = 987

fig = plt.figure()
plt.subplot(2,1,1)
plt.imshow(x_test[index_image], cmap='gray', interpolation='none')
plt.title("Digit: {}".format(y_test[index_image]))

keras.utils.to_categorical(y_test[6],10)



## testing the model
prediction_class=model.predict_classes(np.matrix(x_test[index_image].flatten()))#exemple
prediction_class[0]
