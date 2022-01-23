import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

dataset = pd.read_csv("/Users/yuxizheng/xizheng/proj_past_7007/Week_5/Skin_Cancer_MNIST_HAM10000/hmnist_28_28_RGB.csv")

image_data = dataset.drop(['label'], axis = 1)
image_data = np.array(image_data)
images = image_data.reshape(-1, 28, 28, 3)

from sklearn.model_selection import train_test_split
x_train = []
y_train = []
x_test = []
y_test = []

x_train, x_test, y_train, y_test = train_test_split(image_data, dataset['label'], random_state = 20)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)

def reshape_data(data):
    data_array = np.array(data)
    data = data_array.reshape(-1, 28, 28, 3)
    return data

x_train_balanced = reshape_data(x_train_balanced)
x_train_balanced = x_train_balanced/255
x_test_reshape = reshape_data(x_test)
x_test_reshape = x_test_reshape/255

from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
# define the CNN model
def cnn_model():
    # the input layer
    model = models.Sequential()

    model.add(Conv2D(96, (5, 5), strides=(2, 2), input_shape=(28, 28, 3), activation='relu',padding='valid'))
    model.add(BatchNormalization())  ##keras没有局部归一化，所以用这个代替
    model.add(MaxPooling2D((3, 3), padding='valid',strides=(2, 2)))

    model.add(ZeroPadding2D((2, 2)))
    model.add(Conv2D(256, (5, 5), strides=(1, 1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3),padding='valid', strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(384, (3, 3), activation='relu',padding='same'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D((3, 3), padding='valid',strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    return model

model = cnn_model()

from tensorflow.keras import optimizers
optm = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer= optm, loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])

training = model.fit(x_train_balanced, y_train_balanced, epochs = 25, batch_size= 256, validation_split=0.2, shuffle = True)

model.save('alexnet.h5')

import joblib
joblib.dump(training.history, 'history_alex')

from matplotlib import pyplot as plt
# plot the accuracy of training and validation
plt.plot(training.history['accuracy'])
plt.plot(training.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

# plot the loss of training and validation
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()