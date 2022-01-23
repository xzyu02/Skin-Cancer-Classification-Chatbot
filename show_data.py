# cnn part
import numpy as np
import pandas as pd
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError

# load the model
loaded_model = models.load_model('728cnn.h5')

print(loaded_model.summary())

'''import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# plot the graph of disease distribution in different positions
from matplotlib import pyplot as plt

dataset = pd.read_csv("/Users/yuxizheng/xizheng/proj_past_7007/Week_5/Skin_Cancer_MNIST_HAM10000/hmnist_28_28_RGB.csv")

image_data = dataset.drop(['label'], axis = 1)
image_data = np.array(image_data)
images = image_data.reshape(-1, 28, 28, 3)

plt.figure(figsize = (10,20))
for i in range(5) :
    plt.subplot(1,5,i+1)
    plt.imshow(images[i])
plt.show()'''

'''
import numpy as np
import pandas as pd
from tensorflow.keras import models
import joblib

# load the model
# training = models.load_model("828cnn.h5")
dataset = pd.read_csv("/Users/yuxizheng/xizheng/proj_past_7007/Week_5/Skin_Cancer_MNIST_HAM10000/hmnist_28_28_RGB.csv")
metadata = pd.read_csv("/Users/yuxizheng/xizheng/proj_past_7007/Week_5/Skin_Cancer_MNIST_HAM10000/HAM10000_metadata.csv")
print(metadata['dx'].value_counts())
from matplotlib import pyplot as plt
import seaborn as sns
sns.countplot(x = 'dx', data = metadata)
plt.title('Disease class distribution')
plt.show()
'''
'''
history = joblib.load('/Users/yuxizheng/xizheng/proj_past_7007/Week_9/history_cnn')

print(history['accuracy'])
print(history['val_accuracy'])
print(history['loss'])
print(history['val_loss'])
'''
'''
from matplotlib import pyplot as plt
# plot the accuracy of training and validation
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()

# plot the loss of training and validation
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()
'''
