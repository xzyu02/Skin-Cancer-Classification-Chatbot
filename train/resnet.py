import tensorflow as tf
from tensorflow.keras import layers,activations
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
    data = data_array.reshape((-1, 28, 28, 3)).astype('float32')
    return data


x_train_balanced = reshape_data(x_train_balanced)
x_train_balanced = x_train_balanced/255
x_test_reshape = reshape_data(x_test)
x_test_reshape = x_test_reshape/255

class Residual(tf.keras.Model):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(num_channels,
                                   padding='same',
                                   kernel_size=3,
                                   strides=strides)
        self.conv2 = layers.Conv2D(num_channels, kernel_size=3,padding='same')
        if use_1x1conv:
            self.conv3 = layers.Conv2D(num_channels,
                                       kernel_size=1,
                                       strides=strides)
        else:
            self.conv3 = None
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

    def call(self, X):
        Y = activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return activations.relu(Y + X)

blk = Residual(3)
#tensorflow input shpe     (n_images, x_shape, y_shape, channels).
#mxnet.gluon.nn.conv_layers    (batch_size, in_channels, height, width)
X = tf.random.uniform((4, 6, 6 , 3))
blk(X).shape#TensorShape([4, 6, 6, 3])

blk = Residual(6, use_1x1conv=True, strides=2)
blk(X).shape
#TensorShape([4, 3, 3, 6])
#resnet model
net = tf.keras.models.Sequential(
    [layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
    layers.BatchNormalization(), layers.Activation('relu'),
    layers.MaxPool2D(pool_size=3, strides=2, padding='same')])

class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self,num_channels, num_residuals, first_block=False,**kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.listLayers=[]
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.listLayers.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.listLayers.append(Residual(num_channels))

    def call(self, X):
        for layer in self.listLayers.layers:
            X = layer(X)
        return X

class ResNet(tf.keras.Model):
    def __init__(self,num_blocks,**kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.conv=layers.Conv2D(64, kernel_size=7, strides=2, padding='same')
        self.bn=layers.BatchNormalization()
        self.relu=layers.Activation('relu')
        self.mp=layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.resnet_block1=ResnetBlock(64,num_blocks[0], first_block=True)
        self.resnet_block2=ResnetBlock(128,num_blocks[1])
        self.resnet_block3=ResnetBlock(256,num_blocks[2])
        self.resnet_block4=ResnetBlock(512,num_blocks[3])
        self.gap=layers.GlobalAvgPool2D()
        self.fc=layers.Dense(units=10,activation=tf.keras.activations.softmax)

    def call(self, x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        x=self.mp(x)
        x=self.resnet_block1(x)
        x=self.resnet_block2(x)
        x=self.resnet_block3(x)
        x=self.resnet_block4(x)
        x=self.gap(x)
        x=self.fc(x)
        return x

mynet=ResNet([2,2,2,2])


X = tf.random.uniform(shape=(1,  224, 224 , 3))
for layer in mynet.layers:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)

from tensorflow.keras import optimizers
optm = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

mynet.compile(optimizer= optm, loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])

training = mynet.fit(x_train_balanced, y_train_balanced, epochs = 25, batch_size= 256, validation_split=0.2, shuffle = True)

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

import joblib
joblib.dump(training.history, 'history_ResnetTest')

mynet.save('ResnetTestModel')

test_scores = mynet.evaluate(x_test, y_test, verbose=2)