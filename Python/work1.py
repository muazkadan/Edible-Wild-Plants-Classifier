from typing import List, Any, Union

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import random as rnd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.utils import normalize
import numpy as np

print("Tensorflow version : " + tf.__version__) #2.2.0

IMG_HEIGHT  = int(1920 / 10)
IMG_WIDTH = int(1080 / 10)
DataDir = "/home/mouaz/PycharmProjects/Bitirme/dataset"
CATEGORIES = ["Alfalfa", "Asparagus", "Blue Vervain", "Broadleaf Plantain", "Bull Thistle", "Cattail", "Chickweed",
              "Chicory", "Cleavers", "Coltsfoot", "Common Sow Thistle", "Common Yarrow", "Coneflower",
              "Creeping Charlie",
              "Crimson Clover", "Curly Dock", "Daisy Fleabane", "Dandellion", "Downy Yellow Violet", "Elderberry",
              "Evening Primrose",
              "Fern Leaf Yarrow", "Field Pennycress", "Fireweed", "Forget Me Not", "Garlic Mustard", "Harebell",
              "Henbit", "Herb Robert", "Japanese Knotweed",
              "Joe Pye Weed", "Knapweed", "Kudzu", "Lambs Quarters", "Mallow", "Mayapple", "Meadowsweet",
              "Milk Thistle", "Mullein", "New England Aster",
              "Partridgeberry", "Peppergrass", "Pickerelweed", "Pineapple Weed", "Prickly Pear Cactus",
              "Purple Deadnettle", "Queen Annes Lace",
              "Red Clover", "Sheep Sorrel", "Shepherds Purse", "Spring Beauty", "Sunflower", "Supplejack Vine",
              "Tea Plant", "Teasel", "Toothwort", "Vervain Mallow", "Wild Bee Balm",
              "Wild Black Cherry", "Wild Grape Vine", "Wild Leek", "Wood Sorrel"]

print("Number of categories : " + str(len(CATEGORIES)))

training_data = []
for category in CATEGORIES:
    path = os.path.join(DataDir, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR )
        new_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
        training_data.append([new_array, class_num])

plt.imshow(training_data[1][0])
plt.show()

print(len(training_data))

rnd.shuffle(training_data)

x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 3)

X = np.array(normalize(x))
y = np.array(y)

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(3))
model.add(Activation("sigmoid"))

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X, y, batch_size=32)

