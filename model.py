import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Input, Activation
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, ELU, Lambda

import pandas as pd
import cv2
import matplotlib.image as mimg

def generate_images(file='./data/driving_log_filtered.csv'):
    "Generate input samples from the data directory"
    while 1:
        data = pd.read_csv(file)
        m, _ = data.shape
        file_list = np.random.randint(0, m, m)
        print('generating list')
        for i in file_list:
            angle = data.iloc[i,3]
            
            img = mimg.imread('./data/' + data.iloc[i,0])
            # Crop out the sky and hood, resize the input by 0.50
            img = cv2.resize(img[60:-50,:,:], (0, 0), fx=0.5, fy=0.5)
            #r = [x for x in range(320)]
            #img = cv2.resize(img[60:-50,r[:140] + r[-140:],:], (0,0), fx=0.5, fy=0.5)

            if np.random.random() < 0.5:
                img = cv2.flip(img, 1)
                angle = -1.0 * angle
                
            # Reshape the image into a 4D vector
            imgr = np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
            # Reshape steering angle into a 1-tuple
            if True: #angle > 0.2 or angle < -0.2:
                yield imgr, np.reshape(angle,(1))
            

model = Sequential()
# Normalize the images to fall into (-1, 1) range
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(25,160,3)))

model.add(Conv2D(3,1,1, activation = 'elu')) 

model.add(Conv2D(16, 12, 24)) # subsample=(2, 3)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(.3))

model.add(Conv2D(24, 4, 12, subsample=(2, 2), activation='elu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(.3))

model.add(Flatten())
model.add(Dense(512))
model.add(ELU())
model.add(Dropout(.5))

model.add(Dense(256))
model.add(ELU())
model.add(Dropout(.5))

model.add(Dense(1))
model.summary()

#from keras import models
#model = models.load_model('./model.h5')

# Compile and train the model
model.compile(loss='mse', optimizer='adam')

def save_model(model, filename='model'):
    """Write the model architecture into a JSON file
       Write the model weights intoa HDF5 file"""
       
    with open(filename + '.json','wt') as f:
        f.write(model.to_json())
    
    # Save the model weights into a HDF5 format
    model.save_weights(filename + '.h5')

def model_predict(model, cnt=1):
    """
    Return a table containing model prediction in the first column
    and true value in the second column
    """
    g = generate_images()
    table = []
    for i in range(cnt):
        img = next(g)
        table.append([model.predict(img[0])[0][0], img[1][0]])
    return table
    
    
def get_image_set(n):
    """ 
    Return a tuple containing a set of n input images from the generator 
    and their corresponding labels 
    """
    print('get_image_set called')
    g = generate_images()
    images = np.zeros((n, 25, 160, 3))
    labels = np.zeros(n)
    for i in range(n):
        im, l = next(g)
        images[i,:,:,:] = im[0,:,:,:]
        labels[i] = l
    return images, labels
    
    
# Save the model after each epoch for testing purposes
from keras.callbacks import ModelCheckpoint
import keras
checkpointer = ModelCheckpoint(filepath="./model_check.h5", verbose=1, mode='min', 
                               save_best_only=True,
                               save_weights_only=True)   

# A simple callback class for logging 
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        #print('begin')    

    def on_epoch_end(self, batch, logs={}):
        #self.losses.append(logs.get('loss'))
        print('Prediction after current epoch: ')
        print(model_predict(self.model, 10))   
        pass
        
        
cb = LossHistory()


# Train the model on sample images
#hist = model.fit_generator(generate_images(), 
#                           samples_per_epoch=1000, nb_epoch=10, 
#                           callbacks=[checkpointer,cb], verbose=1)
                           #validation_data=get_val_set(3000))


#from keras import models
#model = models.load_model('model.h5')
model.load_weights('model_check.h5')


import matplotlib.pyplot as plt

for j in range(10):
    print('Starting evaluation {}'.format(j))
    
    val_set = get_image_set(2000)
    
    train_set = get_image_set(20000)
    
    hist = model.fit(train_set[0], train_set[1], batch_size=200,
                               callbacks=[checkpointer,cb], verbose=1,
                               validation_data=val_set)

    save_model(model, 'final_model{}'.format(j))
    
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    
        
    plt.plot(loss)
    plt.title('Training loss')
    plt.show()
    plt.title('Validation loss')
    plt.plot(val_loss)
    plt.show()

    

print('Saving final model')
save_model(model, 'final_model')

print('Final model predictions: ')
print(model_predict(model,20))

