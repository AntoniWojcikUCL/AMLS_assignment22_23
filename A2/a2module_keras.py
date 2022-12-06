import numpy as np
import pandas as pd

# Binary Classification with Sonar Dataset: Standardized
from pandas import read_csv
from keras.models import Sequential
from keras import layers 
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Image manipulation libraries
import cv2

# Constants
DATASET_PATH = './Datasets/celeba'
TEST_DATASET_PATH = './Datasets/celeba_test'
LABEL_IMG_NAMES = "img_name"
LABEL_NAME = "smiling"

def loadImgData(dataset_path, img_names):
    img_data = []

    for i in range(len(img_names)):
        img = cv2.imread(dataset_path + '/img/' + img_names[i])

        h, w, _ = img.shape

        img = img[int(1 * h / 5):int(4 * h / 5), int(2 * w / 8):int(6 * w / 8)]

        # Normalize the image so that values stay low
        img = np.array(img, dtype = np.single)
        
        img_data.append(img)#.flatten())

    img_data = np.array(img_data)

    return img_data

#%%

# load dataset
label_file = pd.read_csv(DATASET_PATH + '/labels.csv', delimiter = "\t")

file_names = label_file[LABEL_IMG_NAMES].values
y = label_file[LABEL_NAME].values

X = loadImgData(DATASET_PATH, file_names)

img_size = 34977

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)

#%%

# baseline model
def create_baseline():
	# create model
	model = Sequential([
	layers.Rescaling(1.0 / 255.0, input_shape=(X.shape[1], X.shape[2], X.shape[3])),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    #layers.Conv2D(32, 3, padding='same', activation='relu'),
    #layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(4, activation='sigmoid'),
    layers.Dense(2)
    ]) 
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

#%%

# evaluate baseline model with standardized dataset
estimators = []

model=create_baseline()
model.summary()

# estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(model=create_baseline, epochs=100, batch_size=-1, random_state = 42)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))