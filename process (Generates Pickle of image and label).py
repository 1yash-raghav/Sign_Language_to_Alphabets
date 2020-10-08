from keras.preprocessing import image
from pathlib import Path
import numpy as np
import random
import pickle

p = Path("./dataset")
dirs = p.glob("*")
image_data = []
labels = []

for folder in p.glob("*"):
    label = str(folder).split('\\')[-1]
    ctr = 0
    present_image_data = []
    for img_path in folder.glob("*.jpg"):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        present_image_data.append(img_array)
    random.shuffle(present_image_data)
    for ix in present_image_data[:50]:
        image_data.append(ix)
        labels.append(label)
    print(label)

X = np.array(image_data)
Y = np.array(labels)
print(np.shape(X), np.shape(Y))

with open('Frames.pickle', 'wb') as handle:
    pickle.dump([image_data, labels], handle, protocol=pickle.HIGHEST_PROTOCOL)
