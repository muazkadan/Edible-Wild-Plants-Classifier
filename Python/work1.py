from typing import List, Any, Union

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import random as rnd

print("Tensorflow version : " + tf.__version__) #2.2.0

IMG_WIDTH = int(1920 / 10)
IMG_HEIGHT = int(1080 / 10)
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
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_UNCHANGED)
        new_array = cv2.resize(img_array, (IMG_HEIGHT, IMG_WIDTH))
        training_data.append([new_array, class_num])

plt.imshow(training_data[1][0])
plt.show()

print(len(training_data))

rnd.shuffle(training_data)