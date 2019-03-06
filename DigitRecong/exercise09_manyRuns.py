import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler


# read train 
train = pd.read_csv("C:/Users/Yotti/Desktop/digit_recongnizer/train.csv")

# read test 
test= pd.read_csv("C:/Users/Yotti/Desktop/digit_recongnizer/test.csv")

# put labels into y_train variable
Y_train = train["label"]
# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1)

# Normalize the data
X_train = X_train / 255.0
test = test / 255.0

# Reshape
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)

Y_train = to_categorical(Y_train, num_classes = 10)



for NumberOfRun in range(1):
    
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print('Number of run: ', NumberOfRun)
    # Split the train and the validation set for the fitting
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
    
    model = Sequential()
    
    model.add(Conv2D(32, (3,3), activation ='relu', input_shape = (28,28,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5,5), strides=2, padding = 'Same', activation ='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    
    model.add(Conv2D(64, (3,3), activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5,5), strides=2, padding = 'Same', activation ='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Conv2D(128, kernel_size = 4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation = "softmax"))
    
    # Compile the model
    model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
    
    # DECREASE LEARNING RATE EACH EPOCH
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
    
    epochs = 45  # for better result increase the epochs
    batch_size = 200
    
    # data augmentation
    datagen = ImageDataGenerator(
            rotation_range=10,  # randomly rotate images in the range 5 degrees
            zoom_range = 0.1, # Randomly zoom image 5%
            width_shift_range=0.1,  # randomly shift images horizontally 5%
            height_shift_range=0.1)  # randomly shift images vertically 5%
    
    datagen.fit(X_train)
    
    # Fit the model
    history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                                  epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)
    
    result=model.predict(test)
    result=np.argmax(result, axis=1)
    
    f=open('C:/Users/Yotti/Desktop/digit_recongnizer/output_0129_'+str(NumberOfRun)+'.csv', 'w')
    f.write('ImageId,Label\n')
    for i in range(len(result)):
        f.write(str(i+1)+','+str(result[i])+'\n')
    f.close()



