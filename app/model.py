from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import tensorflow as tf
import mahotas as mt
from skimage import color
import warnings
warnings.filterwarnings("ignore")

# DeepSat (SAT-6) Data
data_parent_directory = "data/"
train_data_path = data_parent_directory + "X_train_sat6.csv"
train_label_path = data_parent_directory + "y_train_sat6.csv"
test_data_path = data_parent_directory + "X_test_sat6.csv"
test_label_path = data_parent_directory + "y_test_sat6.csv"

# Helper functions
def read_csv_file(data_path, nrows):
    data = pd.read_csv(data_path, header=None, nrows=nrows)
    data = data.values ## converting the data into Numpy array
    return data

def convert_label(data_label):
    labels = []
    for i in range(data_label.shape[0]):
        index = np.nonzero(data_label[i])[0][0]
        if index == 0:
            labels.append("building")
        elif index == 1:
            labels.append("barren land")
        elif index == 2:
            labels.append("tree")
        elif index == 3:
            labels.append("grassland")
        elif index == 4:
            labels.append("road")
        elif index == 5:
            labels.append("water")
    return labels

def feature_extractor(image_file):
    hsv_feature = []
    n = 0
    textFileReader = pd.read_csv(image_file, header=None, chunksize=5000)
    for df_chunk in textFileReader:
        print(n)
        n += 1
        df_chunk = df_chunk.astype("int32")
        data = df_chunk.values
        img = data.reshape(-1,28,28,4)[:, :, :, :3]
        for i in range(len(data)):
            img_hsv = color.rgb2hsv(img[i]) # Image into HSV colorspace
            h = img_hsv[:, :, 0] # Hue
            s = img_hsv[:, :, 1] # Saturation
            v = img_hsv[:, :, 2] # Value aka Lightness
            hsv_feature.append((h.mean(),s.mean(),v.mean()))   
    features = []
    for i in range(len(hsv_feature)):
        h_stack = np.hstack((hsv_feature[i]))
        features.append(h_stack)
            
    return features

print("Initializing feature extraction ...")

train_data_features = feature_extractor(train_data_path)
feature_train = pd.DataFrame(train_data_features, columns=["feature"+ str(i) for i in range(len(train_data_features[0]))])
feature_train.to_csv("train_feature.csv")

test_data_features = feature_extractor(test_data_path)
feature_test = pd.DataFrame(test_data_features, columns=["feature"+ str(i) for i in range(len(test_data_features[0]))])
feature_test.to_csv("test_feature.csv")

print("Finishing feature extraction ...")

def read_feature(feature_path):
    df = pd.read_csv(feature_path, index_col=[0])
    return df

def read_label(label_path):
    df = pd.read_csv(label_path, header=None)
    return df

train_feature = read_feature("train_feature.csv")
train_label = read_label(label_path=train_label_path)
test_feature = read_feature("test_feature.csv")
test_label = read_label(label_path=test_label_path)

sc = StandardScaler()
#fit the training data
fit = sc.fit(train_feature)
train_data_stn = fit.transform(train_feature)
test_data_stn = fit.transform(test_feature)

#### MODEL ####
model = Sequential()

# layer1
model.add(Dense(units=50,input_shape=(train_data_stn.shape[1],),use_bias=True))
model.add(Activation("relu"))
model.add(Dropout(0.2))

# layer2
model.add(Dense(units=50, use_bias=True))
model.add(Activation("relu"))
model.add(Dropout(0.2))

# layer3
model.add(Dense(units=6, activation="softmax"))

# ADD early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

print("Building the model ...")

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(train_data_stn, train_label.values, validation_split=0.15, batch_size=512, epochs=20,callbacks=[es])

print("Model built ...")

accuracy_on_test_data = model.evaluate(test_data_stn, test_label.values)[1]

print("Accuracy on test data: ", accuracy_on_test_data)

model.save("my_model_v2", save_format="h5")