import numpy as np
import math

FILENAME = "1.txt"

lines = []
with open(FILENAME) as f:
    lines = [x.strip() for x in f.readlines()]

print(lines)
# Parse
data = []
labels = []
for l in lines:
    sip, dip, label = l.split(" ")
    data.append((sip,dip))
    labels.append(label)

print(data[:5])
print(labels[:5])

# Building detection windows
WINDOW = 10
PROP = 0.2
windows = []
windowlabels = []
for i in range(len(data)-WINDOW):
    windows.append(data[i:i+WINDOW])
    windowlabels.append(labels[i:i+WINDOW])
for i in range(len(windowlabels)):
    if windowlabels[i].count("DOS") > WINDOW*PROP:
        windowlabels[i] = 1
    else:
        windowlabels[i] = 0

print(windows[:5])
print(windowlabels[:5])

#Shuffling everything
temp = np.array(list(zip(windows, windowlabels)))
np.random.shuffle(temp)
allData = [x[0] for x in temp]
allLabels = [x[1] for x in temp]

# Parsing the data into ML usable form
# will be turning the windows into a single row of src and dst pairs.
# first we have to bin encode by taking all unique ip addresses
class binarizer:
    def __init__(self, allData):
        self.ref = list(set(allData))
        self.num = math.ceil(math.log2(len(self.ref)))
    def transform(self, data):
        i = self.ref.index(data)
        b = bin(i)[2:]
        return [0 for i in range(self.num-len(b))] + [int(x) for x in b]

allIPS = []
for l in lines:
    sip, dip, label = l.split(" ")
    allIPS.append(sip)
    allIPS.append(dip)
b = binarizer(allIPS)

if False: # For using windows
    normData = []
    for w in windows:
        normW = []
        for sip, dip in w:
            s = [np.float(x) for x in b.transform(sip)]
            d = [np.float(x) for x in b.transform(sip)]
            normW.append(s+d)
        normData.append(np.array(normW))

if True:
    temp = np.array(list(zip(data, labels)))
    np.random.shuffle(temp)
    allData = [x[0] for x in temp]
    allLabels = [x[1] for x in temp]

    n = []
    for l in allLabels:
        if l == "DOS":
            n.append(1.0)
        else:
            n.append(0.0)
    allLabels = n

    normData = []
    for sip, dip in allData:
        s = [np.float(x) for x in b.transform(sip)]
        d = [np.float(x) for x in b.transform(sip)]
        normData.append(s+d)

# Training and Evaluation
print(f"Total data {len(normData)}")
print(f"Total labels {len(allLabels)}")
TRAIN_RATIO = 0.9
TRAIN_LEN = int(len(normData)*TRAIN_RATIO)
training_data = normData[:TRAIN_LEN]
training_labels = allLabels[:TRAIN_LEN]
testing_data = normData[TRAIN_LEN:]
testing_labels = allLabels[TRAIN_LEN:]
print(f"Training data len {len(training_data)} label len {len(training_labels)}")
print(f"Testing data len {len(testing_data)} label len {len(testing_labels)}")

# x,y = training_data[0].shape
# training_data = np.array(training_data).reshape(-1,1,x,y)
# training_labels = np.array(training_labels)
# testing_data = np.array(testing_data).reshape(-1,1,x,y)
# testing_labels = np.array(testing_labels)
# training_data = np.array(training_data).reshape(-1,x,y)
# training_labels = np.array(training_labels)
# testing_data = np.array(testing_data).reshape(-1,x,y)
# testing_labels = np.array(testing_labels)
training_data = np.array(training_data).reshape(-1,1,len(normData[0]))
training_labels = np.array(training_labels)
testing_data = np.array(testing_data).reshape(-1,1,len(normData[0]))
testing_labels = np.array(testing_labels)
print(f"Shape of training data is {training_data.shape}")
print(f"Shape of testing data is {testing_data.shape}")
print(f"First value in training is {training_data[0]}")
print(f"First 10 training labels {training_labels[:10]}")
print(f"First value in testing is {testing_data[0]}")
print(f"First 10 testing labels {testing_labels[:10]}")
print(f"Training on {np.count_nonzero(training_labels==1)} and {np.count_nonzero(training_labels==0)} Benign")
print(f"Testing on {np.count_nonzero(testing_labels==1)} and {np.count_nonzero(testing_labels==0)} Benign")

# print(training_data[0][0].shape)
# for d in normData:
#     print(len(d))
#     print(d)

EPOCHS = 100

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM#, CuDNNLSTM

model = Sequential()
# IF you are running with a GPU, try out the CuDNNLSTM layer type instead (don't pass an activation, tanh is required)
model.add(LSTM(10, activation='relu', return_sequences=True))
# model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
# model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

model.fit(training_data,
        training_labels,
        epochs=EPOCHS,
        validation_data=(testing_data, testing_labels))

