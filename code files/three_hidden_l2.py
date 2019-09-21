import numpy
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import keras
import keras.backend as K
from keras.callbacks import LambdaCallback
from keras import regularizers
from keras.regularizers import l2



seed = 7
numpy.random.seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train= X_train.reshape(60000,784)
#X_test= X_test.reshape(60000,784)


# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

#for i in range(0,9):
X0=np.zeros(shape=(100,784))
X00=X_train[y_train[:,]==0]
X0=X00[0:100,:]

X1=np.zeros(shape=(100,784))
X11=X_train[y_train[:,]==1]
X1=X11[0:100,:]

X2=np.zeros(shape=(100,784))
X22=X_train[y_train[:,]==2]
X2=X22[0:100,:]

X3=np.zeros(shape=(100,784))
X33=X_train[y_train[:,]==3]
X3=X33[0:100,:]

X4=np.zeros(shape=(100,784))
X44=X_train[y_train[:,]==4]
X4=X44[0:100,:]

X5=np.zeros(shape=(100,784))
X55=X_train[y_train[:,]==5]
X5=X55[0:100,:]

X6=np.zeros(shape=(100,784))
X66=X_train[y_train[:,]==6]
X6=X66[0:100,:]

X7=np.zeros(shape=(100,784))
X77=X_train[y_train[:,]==7]
X7=X77[0:100,:]

X8=np.zeros(shape=(100,784))
X88=X_train[y_train[:,]==8]
X8=X88[0:100,:]

X9=np.zeros(shape=(100,784))
X99=X_train[y_train[:,]==9]
X9=X99[0:100,:]

#X_1000_train=[np.zeros(shape=(1000,784))
X_train = np.concatenate((X0,X1,X2,X3,X4,X5,X6,X7,X8,X9))


X0t=np.zeros(shape=(100,784))
X00t=X_test[y_test[:,]==0]
X0t=X00t[0:100,:]

X1t=np.zeros(shape=(100,784))
X11t=X_test[y_test[:,]==1]
X1t=X11t[0:100,:]

X2t=np.zeros(shape=(100,784))
X22t=X_test[y_test[:,]==2]
X2t=X22t[0:100,:]

X3t=np.zeros(shape=(100,784))
X33t=X_test[y_test[:,]==3]
X3t=X33t[0:100,:]

X4t=np.zeros(shape=(100,784))
X44t=X_test[y_test[:,]==4]
X4t=X44t[0:100,:]

X5t=np.zeros(shape=(100,784))
X55t=X_test[y_test[:,]==5]
X5t=X55t[0:100,:]

X6t=np.zeros(shape=(100,784))
X66t=X_test[y_test[:,]==6]
X6t=X66t[0:100,:]

X7t=np.zeros(shape=(100,784))
X77t=X_test[y_test[:,]==7]
X7t=X77t[0:100,:]

X8t=np.zeros(shape=(100,784))
X88t=X_test[y_test[:,]==8]
X8t=X88t[0:100,:]

X9t=np.zeros(shape=(100,784))
X99t=X_test[y_test[:,]==9]
X9t=X99t[0:100,:]

#X_1000_train=[np.zeros(shape=(1000,784))

X_test = np.concatenate((X0t,X1t,X2t,X3t,X4t,X5t,X6t,X7t,X8t,X9t))

y_test = np.zeros(1000,)

for i in range(0,10):
    for j in range(0,100):
        y_test[(100*i)+j,]=i
  
y_train = np.zeros(1000,)

for i in range(0,10):
    for j in range(0,100):
        y_train[(100*i)+j,]=i


# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

learning_rate=[]
class SGDLearningRateTracker(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        _lr = tf.to_float(optimizer.lr, name='ToFloat')
        _decay = tf.to_float(optimizer.decay, name='ToFloat')
        _iter = tf.to_float(optimizer.iterations, name='ToFloat')
        
        lr = K.eval(_lr * (1. / (1. + _decay * _iter)))
        learning_rate.append(lr)
        print(' - LR: {:.6f}\n'.format(lr))
#collecting test loss and accuracy in an array     
loss_collected_test=[]
acc_collected_test=[]
class TestCallback_test(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        loss_collected_test.append(loss)
        acc_collected_test.append(acc)
        
#collecting train loss and accuracy in an array     
loss_collected_train=[]
acc_collected_train=[]
class TestCallback_train(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        loss_collected_train.append(loss)
        acc_collected_train.append(acc)
        
# define baseline model
def baseline_model():
	# create model
    model = Sequential()
    model.add(Dense(30, input_dim=num_pixels, kernel_initializer='normal', activation='sigmoid',W_regularizer=l2(5)))
    model.add(Dense(30, kernel_initializer='normal', activation='sigmoid',W_regularizer=l2(5)))
    model.add(Dense(30, kernel_initializer='normal', activation='sigmoid',W_regularizer=l2(5)))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1,decay=0.0001), metrics=['accuracy'])
    return model

# build the model
model = baseline_model()
# Fit the model

print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))

callbacks_list = [SGDLearningRateTracker(),TestCallback_test((X_test, y_test)),TestCallback_train((X_train, y_train))]


history=model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=30, batch_size=10,
verbose=2,callbacks = callbacks_list)


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Training/testing data accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('criterion function on training/testing data set')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#plotting learning speed 
Epoch_graph=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
plt.plot(Epoch_graph,learning_rate)
plt.xlabel('Epoch')
plt.ylabel('Learning Speed')
plt.title('learning speed of the hidden layer')
plt.show()

#plotting errors
#test error
plt.plot(Epoch_graph,loss_collected_test)
plt.xlabel('Epoch')
plt.ylabel('Test_error')
plt.title('Test Error')
plt.show()

#train error
plt.plot(Epoch_graph,loss_collected_train)
plt.xlabel('Epoch')
plt.ylabel('Train_error')
plt.title('Train Error')
plt.show()



