from data_process import *

from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.model_selection import train_test_split
from keras.layers import *

def behavioral_cloning_model():

    # set up lambda layer to normalize and mean center the data
    model = Sequential()
    # set up lambda layer to normalize and mean center the data
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # crop the upper part of the image which include (trees, sky, etc.) and lower part of the image which includes the car's hood
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    # Model architecture
    model.add(Conv2D(filters=24, kernel_size=5, strides=(2,2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=5, strides=(2,2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=5, strides=(2,2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss = 'mse', optimizer='adam')
    
    return model


csv_file = '../../opt/carnd_p3/data/driving_log.csv'
img_path = '../../opt/carnd_p3/data/'
batch_size = 32

#read the csv file and add them in csv_lines
csv_lines = csv_reader(csv_file)
train_samples, validation_samples = train_test_split(csv_lines, test_size=0.2)

train_generator = images_generator(train_samples, img_path, batch_size=batch_size)
validation_generator = images_generator(validation_samples, img_path, batch_size=batch_size)


model = behavioral_cloning_model()

model.fit_generator(train_generator, \
            steps_per_epoch=np.ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=np.ceil(len(validation_samples)/batch_size), \
            epochs=10, verbose=1)

model.save("model.h5")